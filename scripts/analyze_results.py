"""
Analyze scored results and produce tables and charts.

Outputs:
- data/analysis/overall_ranking.csv     -> Overall LLM rankings
- data/analysis/per_task_ranking.csv    -> Rankings broken down by task type
- data/analysis/per_level_ranking.csv   -> Rankings by DAG difficulty level
- data/analysis/heatmap.png             -> Heatmap of LLM x task scores
- data/analysis/bar_overall.png         -> Bar chart of overall scores
- data/analysis/bar_per_task.png        -> Grouped bar chart per task
"""

import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_all_scores(scores_dir):
    """Load and consolidate all latest score files into a flat list of records."""
    # Find latest score file per LLM
    latest = {}
    for filepath in scores_dir.glob("*_scores_*.json"):
        parts = filepath.stem.replace("_scores", "").rsplit("_", 2)
        if len(parts) >= 3:
            llm_name = parts[0]
            timestamp = "_".join(parts[1:])
            if llm_name not in latest or timestamp > latest[llm_name][1]:
                latest[llm_name] = (filepath, timestamp)

    # Build flat records: one row per (llm, dag, task, metric)
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
                    "level": int(s["dag_id"][1]),  # L1 -> 1, L2 -> 2, L3 -> 3
                })
    return pd.DataFrame(records)


def overall_ranking(df, output_path):
    """Compute overall mean score per LLM."""
    ranking = df.groupby("llm")["score"].agg(["mean", "std", "count"]).round(3)
    ranking = ranking.sort_values("mean", ascending=False)
    ranking.columns = ["mean_score", "std_dev", "n_data_points"]
    ranking.to_csv(output_path)
    print(f"\n=== Overall Ranking ===")
    print(ranking)
    return ranking


def per_task_ranking(df, output_path):
    """Mean score per LLM per task type."""
    pivot = df.groupby(["llm", "task_type"])["score"].mean().round(3).unstack()
    pivot["overall"] = pivot.mean(axis=1).round(3)
    pivot = pivot.sort_values("overall", ascending=False)
    pivot.to_csv(output_path)
    print(f"\n=== Per-Task Ranking ===")
    print(pivot)
    return pivot


def per_level_ranking(df, output_path):
    """Mean score per LLM per difficulty level."""
    pivot = df.groupby(["llm", "level"])["score"].mean().round(3).unstack()
    pivot.columns = [f"L{c}" for c in pivot.columns]
    pivot["overall"] = pivot.mean(axis=1).round(3)
    pivot = pivot.sort_values("overall", ascending=False)
    pivot.to_csv(output_path)
    print(f"\n=== Per-Level Ranking ===")
    print(pivot)
    return pivot


def heatmap_plot(df, output_path):
    """Heatmap of LLM × task showing mean scores."""
    pivot = df.groupby(["llm", "task_type"])["score"].mean().unstack()
    
    # Sort LLMs by overall score
    overall = pivot.mean(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[overall.index]
    
    plt.figure(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=1,
        vmax=5,
        cbar_kws={"label": "Mean score (1-5)"},
    )
    plt.title("LLM Performance by Task Type")
    plt.ylabel("LLM")
    plt.xlabel("Task Type")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path.name}")


def bar_overall_plot(df, output_path):
    """Horizontal bar chart of overall scores."""
    means = df.groupby("llm")["score"].mean().sort_values()
    
    plt.figure(figsize=(8, 5))
    bars = plt.barh(means.index, means.values, color="steelblue")
    plt.xlabel("Mean Score (1-5)")
    plt.title("Overall LLM Performance on Causal Reasoning Tasks")
    plt.xlim(0, 5)
    
    # Add value labels at end of each bar
    for bar, val in zip(bars, means.values):
        plt.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path.name}")


def bar_per_task_plot(df, output_path):
    """Grouped bar chart showing each LLM's score per task type."""
    pivot = df.groupby(["llm", "task_type"])["score"].mean().unstack()
    
    # Sort LLMs by overall score (highest first)
    overall = pivot.mean(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[overall.index]
    
    pivot.plot(kind="bar", figsize=(11, 6), colormap="viridis", width=0.8)
    plt.ylabel("Mean Score (1-5)")
    plt.xlabel("LLM")
    plt.title("LLM Performance Across Task Types")
    plt.ylim(0, 5)
    plt.legend(title="Task Type")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path.name}")


def main():
    project_root = Path(__file__).parent.parent
    scores_dir = project_root / "data" / "scores"
    analysis_dir = project_root / "data" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scores from: {scores_dir}")
    df = load_all_scores(scores_dir)
    print(f"Loaded {len(df)} score data points across {df['llm'].nunique()} LLMs")

    print("\n--- Generating tables ---")
    overall_ranking(df, analysis_dir / "overall_ranking.csv")
    per_task_ranking(df, analysis_dir / "per_task_ranking.csv")
    per_level_ranking(df, analysis_dir / "per_level_ranking.csv")

    print("\n--- Generating charts ---")
    heatmap_plot(df, analysis_dir / "heatmap.png")
    bar_overall_plot(df, analysis_dir / "bar_overall.png")
    bar_per_task_plot(df, analysis_dir / "bar_per_task.png")

    print(f"\n✅ Analysis complete. Outputs in: {analysis_dir}")


if __name__ == "__main__":
    main()