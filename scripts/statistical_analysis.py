"""
Statistical significance testing for LLM rankings.

Tests:
1. Friedman test: is there any significant difference among the 6 LLMs?
2. Mann-Whitney U pairwise: which specific pairs differ significantly?
3. Cliff's delta: effect size for each pairwise comparison

Outputs:
- data/analysis/friedman_test.txt
- data/analysis/pairwise_significance.csv
- data/analysis/effect_sizes.csv
"""

import json
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
from scipy import stats


def load_all_scores(scores_dir):
    """Load latest score files into a flat DataFrame."""
    latest = {}
    for filepath in scores_dir.glob("*_scores_*.json"):
        parts = filepath.stem.replace("_scores", "").rsplit("_", 2)
        if len(parts) >= 3:
            llm_name = parts[0]
            timestamp = "_".join(parts[1:])
            if llm_name not in latest or timestamp > latest[llm_name][1]:
                latest[llm_name] = (filepath, timestamp)

    records = []
    for llm_name, (filepath, _) in latest.items():
        with open(filepath) as f:
            data = json.load(f)
        for s in data.get("scores", []):
            scores = s.get("scores")
            if not scores:
                continue
            for metric, value in scores.items():
                records.append({
                    "llm": llm_name,
                    "dag_id": s["dag_id"],
                    "task_type": s["task_type"],
                    "metric": metric,
                    "score": value,
                })
    return pd.DataFrame(records)


def cliffs_delta(x, y):
    """
    Compute Cliff's delta — effect size for ordinal data.
    Returns a value in [-1, 1]:
        positive = x tends to be larger than y
        negative = y tends to be larger than x
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    
    # Count pairs where x > y, x < y, x == y
    greater = sum((xi > y).sum() for xi in x)
    less = sum((xi < y).sum() for xi in x)
    
    delta = (greater - less) / (n_x * n_y)
    return delta


def interpret_delta(delta):
    """Interpret magnitude of Cliff's delta."""
    abs_d = abs(delta)
    if abs_d < 0.147:
        return "negligible"
    elif abs_d < 0.33:
        return "small"
    elif abs_d < 0.474:
        return "medium"
    else:
        return "large"


def friedman_test(df, output_path):
    """
    Test whether at least one LLM differs from the others.
    Each (dag_id, task_type, metric) combination is treated as one 'block'.
    """
    # Pivot so each row is a block, each column is an LLM
    df["block"] = df["dag_id"] + "_" + df["task_type"] + "_" + df["metric"]
    pivot = df.pivot_table(
        index="block",
        columns="llm",
        values="score",
        aggfunc="mean"
    )
    
    # Drop rows where any LLM is missing (Friedman needs complete data)
    pivot_complete = pivot.dropna()
    
    # Each column becomes a separate array
    score_arrays = [pivot_complete[col].values for col in pivot_complete.columns]
    
    statistic, p_value = stats.friedmanchisquare(*score_arrays)
    
    result_text = f"""=== Friedman Test ===
H0: All LLMs perform equally
H1: At least one LLM differs significantly

Sample blocks (complete cases): {len(pivot_complete)}
LLMs compared: {list(pivot_complete.columns)}

Test statistic: {statistic:.4f}
p-value: {p_value:.6f}

Interpretation:
{'✓ SIGNIFICANT — at least one LLM differs (p < 0.05)' if p_value < 0.05 else '✗ NOT significant — no clear differences'}
{'✓ HIGHLY SIGNIFICANT (p < 0.001)' if p_value < 0.001 else ''}
"""
    
    with open(output_path, "w") as f:
        f.write(result_text)
    
    print(result_text)
    return statistic, p_value


def pairwise_mann_whitney(df, output_path):
    """Pairwise Mann-Whitney U tests with Bonferroni correction."""
    llms = sorted(df["llm"].unique())
    pairs = list(combinations(llms, 2))
    n_comparisons = len(pairs)
    bonferroni_alpha = 0.05 / n_comparisons
    
    results = []
    for llm_a, llm_b in pairs:
        scores_a = df[df["llm"] == llm_a]["score"].values
        scores_b = df[df["llm"] == llm_b]["score"].values
        
        # Two-sided Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            scores_a, scores_b, alternative="two-sided"
        )
        
        # Direction: which LLM scored higher (mean comparison)
        mean_a = scores_a.mean()
        mean_b = scores_b.mean()
        winner = llm_a if mean_a > mean_b else llm_b
        
        # Significance flag
        significant_uncorrected = p_value < 0.05
        significant_bonferroni = p_value < bonferroni_alpha
        
        results.append({
            "llm_a": llm_a,
            "llm_b": llm_b,
            "mean_a": round(mean_a, 3),
            "mean_b": round(mean_b, 3),
            "winner": winner,
            "U_statistic": round(statistic, 1),
            "p_value": round(p_value, 6),
            "significant_p<0.05": significant_uncorrected,
            "significant_bonferroni": significant_bonferroni,
        })
    
    df_results = pd.DataFrame(results).sort_values("p_value")
    df_results.to_csv(output_path, index=False)
    
    print(f"\n=== Pairwise Mann-Whitney U Tests ===")
    print(f"Bonferroni-corrected α: {bonferroni_alpha:.5f}")
    print(df_results.to_string(index=False))
    return df_results


def effect_sizes(df, output_path):
    """Compute Cliff's delta for each LLM pair."""
    llms = sorted(df["llm"].unique())
    pairs = list(combinations(llms, 2))
    
    results = []
    for llm_a, llm_b in pairs:
        scores_a = df[df["llm"] == llm_a]["score"].values
        scores_b = df[df["llm"] == llm_b]["score"].values
        
        delta = cliffs_delta(scores_a, scores_b)
        
        results.append({
            "llm_a": llm_a,
            "llm_b": llm_b,
            "cliffs_delta": round(delta, 3),
            "magnitude": interpret_delta(delta),
            "interpretation": (
                f"{llm_a} > {llm_b}" if delta > 0
                else f"{llm_b} > {llm_a}"
            ) + f" ({interpret_delta(delta)} effect)"
        })
    
    df_results = pd.DataFrame(results).sort_values(
        "cliffs_delta", key=abs, ascending=False
    )
    df_results.to_csv(output_path, index=False)
    
    print(f"\n=== Effect Sizes (Cliff's delta) ===")
    print(df_results.to_string(index=False))
    return df_results


def main():
    project_root = Path(__file__).parent.parent
    scores_dir = project_root / "data" / "scores"
    analysis_dir = project_root / "data" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scores from: {scores_dir}")
    df = load_all_scores(scores_dir)
    print(f"Loaded {len(df)} score data points across {df['llm'].nunique()} LLMs\n")

    # Test 1: Overall — is there ANY difference?
    friedman_test(df, analysis_dir / "friedman_test.txt")

    # Test 2: Pairwise — which specific pairs differ?
    pairwise_mann_whitney(df, analysis_dir / "pairwise_significance.csv")

    # Test 3: Effect size — how big are the differences?
    effect_sizes(df, analysis_dir / "effect_sizes.csv")

    print(f"\n✅ Statistical analysis complete. Outputs in: {analysis_dir}")


if __name__ == "__main__":
    main()