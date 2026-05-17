"""
Compute variance metrics across consistency study trials.
Outputs = a CSV ready.
"""

import json
import statistics
from pathlib import Path

import pandas as pd


def analyze_consistency():
    project_root = Path(__file__).parent.parent
    consistency_dir = project_root / "data" / "consistency"
    
    # Find latest scored file
    score_files = sorted(consistency_dir.glob("consistency_scores_*.json"))
    if not score_files:
        print("❌ No scored consistency files found.")
        return
    
    latest_file = score_files[-1]
    with open(latest_file) as f:
        data = json.load(f)
    
    # Build a flat DataFrame: one row per (llm, dag, task, trial, mean_score)
    records = []
    for s in data["scored_trials"]:
        if s.get("scores"):
            mean_score = sum(s["scores"].values()) / len(s["scores"])
            records.append({
                "llm": s["llm"],
                "dag_id": s["dag_id"],
                "task_type": s["task_type"],
                "trial": s["trial"],
                "mean_score": mean_score,
            })
    
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} trial records")
    
    # Group by (llm, dag, task) and compute variance across trials
    grouped = df.groupby(["llm", "dag_id", "task_type"])["mean_score"].agg([
        ("mean", "mean"),
        ("std", "std"),
        ("min", "min"),
        ("max", "max"),
        ("range", lambda x: x.max() - x.min()),
        ("n_trials", "count"),
    ]).round(3).reset_index()
    
    # Save the per-condition table
    output_path = project_root / "data" / "analysis"
    output_path.mkdir(parents=True, exist_ok=True)
    
    grouped.to_csv(output_path / "consistency_per_condition.csv", index=False)
    print(f"\n=== Per-condition variance ===")
    print(grouped.to_string(index=False))
    
    # Compute summary statistics across all conditions
    print(f"\n=== Summary statistics ===")
    overall_std = grouped["std"].mean()
    overall_range = grouped["range"].mean()
    max_std = grouped["std"].max()
    
    print(f"Mean within-condition standard deviation: {overall_std:.3f}")
    print(f"Mean within-condition range:              {overall_range:.3f}")
    print(f"Largest standard deviation:               {max_std:.3f}")
    
    # Per-LLM breakdown — useful for thesis
    per_llm = grouped.groupby("llm")["std"].agg(["mean", "max"]).round(3)
    print(f"\n=== Per-LLM variance ===")
    print(per_llm.to_string())
    per_llm.to_csv(output_path / "consistency_per_llm.csv")
    
    # Per-task breakdown
    per_task = grouped.groupby("task_type")["std"].agg(["mean", "max"]).round(3)
    print(f"\n=== Per-task variance ===")
    print(per_task.to_string())
    per_task.to_csv(output_path / "consistency_per_task.csv")
    
    print(f"\n✅ Analysis complete. CSVs saved to: {output_path}")


if __name__ == "__main__":
    analyze_consistency()