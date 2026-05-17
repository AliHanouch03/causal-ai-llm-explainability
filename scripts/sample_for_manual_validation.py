"""
Randomly sample 30 LLM responses for manual validation.
Outputs a CSV that the human can fill in by hand, then a comparison
script will compute inter-rater agreement against the LLM judge.
"""

import json
import random
import csv
from pathlib import Path


# Configuration
SAMPLE_SIZE = 30
RANDOM_SEED = 42  # For reproducibility


def sample_responses(results_dir="data/results", output_file="data/manual_validation/sample.csv"):
    project_root = Path(__file__).parent.parent
    results_path = project_root / results_dir
    output_path = project_root / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all successful responses from all LLM result files
    all_responses = []
    
    for filepath in results_path.glob("*.json"):
        # Take only the most recent file per LLM
        with open(filepath) as f:
            data = json.load(f)
        
        llm_name = data["llm"]
        for result in data["results"]:
            if result.get("error") or not result.get("response"):
                continue
            all_responses.append({
                "llm": llm_name,
                "dag_id": result["dag_id"],
                "task_type": result["task_type"],
                "response": result["response"],
            })
    
    # Stratified sample: balanced across task types
    random.seed(RANDOM_SEED)
    by_task = {"explanation": [], "counterfactual": [], "critique": []}
    for r in all_responses:
        if r["task_type"] in by_task:
            by_task[r["task_type"]].append(r)
    
    per_task = SAMPLE_SIZE // 3
    sampled = []
    for task, items in by_task.items():
        if len(items) >= per_task:
            sampled.extend(random.sample(items, per_task))
        else:
            sampled.extend(items)
    
    # Write to CSV with empty score columns for manual filling
    metric_columns = {
        "explanation": ["structural_accuracy", "pattern_recognition", "completeness", "clarity", "faithfulness"],
        "counterfactual": ["direction_correctness", "scope_correctness", "mechanism_explanation", "causal_reasoning"],
        "critique": ["flaw_detection", "flaw_classification", "false_positive_avoidance", "explanation_quality"],
    }
    
    # Get all possible metric names
    all_metrics = sorted(set(m for metrics in metric_columns.values() for m in metrics))
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(
            ["sample_id", "llm", "dag_id", "task_type", "response_preview", "response_full"]
            + all_metrics
            + ["notes"]
        )
        # Rows
        for i, r in enumerate(sampled):
            preview = r["response"][:200].replace("\n", " ") + "..."
            row = [
                i + 1,
                r["llm"],
                r["dag_id"],
                r["task_type"],
                preview,
                r["response"],
            ]
            # Empty cells for manual scoring 
            relevant_metrics = set(metric_columns[r["task_type"]])
            for m in all_metrics:
                row.append("" if m in relevant_metrics else "N/A")
            row.append("")  # notes
            writer.writerow(row)
    
    print(f"✅ Sampled {len(sampled)} responses to: {output_path}")


if __name__ == "__main__":
    sample_responses()