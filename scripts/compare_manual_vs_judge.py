"""
Compare manual scores against the LLM judge's scores.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict


def load_judge_scores(scores_dir):
    """Build lookup: (llm, dag_id, task_type) -> judge scores dict."""
    judge_lookup = {}
    for filepath in scores_dir.glob("*_scores_*.json"):
        with open(filepath) as f:
            data = json.load(f)
        llm = data["llm"]
        for s in data["scores"]:
            if s.get("scores"):
                key = (llm, s["dag_id"], s["task_type"])
                judge_lookup[key] = s["scores"]
    return judge_lookup


def load_manual_scores(sample_file):
    """Load manually-filled scores from CSV."""
    excluded_cols = {"sample_id", "llm", "dag_id", "task_type", "response_preview", "response_full", "notes"}
    
    manual = []
    with open(sample_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores = {}
            for k, v in row.items():
                if k in excluded_cols:
                    continue
                if v and v.strip() and v != "N/A":
                    try:
                        scores[k] = int(v.strip())
                    except ValueError:
                        continue
            if scores:
                manual.append({
                    "llm": row["llm"],
                    "dag_id": row["dag_id"],
                    "task_type": row["task_type"],
                    "scores": scores,
                })
    return manual


def compute_agreement(manual, judge_lookup):
    """Compute inter-rater agreement statistics."""
    diffs = []
    exact_match = 0
    within_one = 0
    total = 0
    per_metric = defaultdict(list)
    per_task = defaultdict(list)

    unmatched = []
    for m in manual:
        key = (m["llm"], m["dag_id"], m["task_type"])
        if key not in judge_lookup:
            unmatched.append(key)
            continue
        judge_scores = judge_lookup[key]
        for metric, manual_val in m["scores"].items():
            if metric in judge_scores:
                judge_val = judge_scores[metric]
                diff = abs(manual_val - judge_val)
                diffs.append(diff)
                per_metric[metric].append(diff)
                per_task[m["task_type"]].append(diff)
                if diff == 0:
                    exact_match += 1
                if diff <= 1:
                    within_one += 1
                total += 1

    if total == 0:
        print("❌ No matched scores to compare. Maybe you didn't fill in the CSV!")
        return

    print(f"\n=== Inter-rater Agreement (Manual vs LLM Judge) ===")
    print(f"Total cells compared:       {total}")
    print(f"Mean absolute difference:   {sum(diffs)/total:.2f}")
    print(f"Exact match rate:           {exact_match/total*100:.1f}%")
    print(f"Within 1 point rate:        {within_one/total*100:.1f}%")
    
    if unmatched:
        print(f"\n!!!  {len(unmatched)} manual entries had no matching judge score")

    print(f"\n--- Per-metric breakdown ---")
    for metric, vals in sorted(per_metric.items()):
        avg = sum(vals) / len(vals)
        print(f"  {metric:<32} mean diff={avg:.2f}  (n={len(vals)})")
    
    print(f"\n--- Per-task breakdown ---")
    for task, vals in sorted(per_task.items()):
        avg = sum(vals) / len(vals)
        print(f"  {task:<32} mean diff={avg:.2f}  (n={len(vals)})")


def main():
    project_root = Path(__file__).parent.parent
    scores_dir = project_root / "data" / "scores"
    sample_file = project_root / "data" / "manual_validation" / "sample.csv"
    
    judge = load_judge_scores(scores_dir)
    manual = load_manual_scores(sample_file)
    
    print(f"Loaded {len(manual)} manual entries, {len(judge)} judge scores")
    compute_agreement(manual, judge)


if __name__ == "__main__":
    main()