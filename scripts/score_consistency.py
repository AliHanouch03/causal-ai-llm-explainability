"""
Score the consistency study trials using Nemotron.
Outputs scores per-trial so we can compute variance across repeats.

Includes resume logic so if ever interrupted, re-running will skip already-scored trials.
-> Important for token limit issues
"""

import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv(Path(__file__).parent.parent / "src" / ".env")

from dag_loader import load_all_dags
from llm_clients import get_client

# Reuse functions from score_results
sys.path.insert(0, str(Path(__file__).parent))
from score_results import (
    build_judge_prompt,
    parse_judge_response,
    score_with_retry,
    METRICS,
)


JUDGE_MODEL = "nemotron-3-super"


def load_existing_scores(consistency_dir):
    """
    Load all successfully scored trials from any previous score runs.
    Returns dict keyed by (llm, dag_id, task_type, trial) -> scored entry.
    """
    existing = {}
    for filepath in consistency_dir.glob("consistency_scores_*.json"):
        try:
            with open(filepath) as f:
                old_data = json.load(f)
            for s in old_data.get("scored_trials", []):
                scores = s.get("scores")
                if (scores is not None
                    and isinstance(scores, dict)
                    and len(scores) > 0):
                    key = (s["llm"], s["dag_id"], s["task_type"], s["trial"])
                    existing[key] = s
        except Exception:
            continue
    return existing


def score_consistency():
    project_root = Path(__file__).parent.parent
    consistency_dir = project_root / "data" / "consistency"

    # Find latest trials file
    trial_files = sorted(consistency_dir.glob("consistency_trials_*.json"))
    if not trial_files:
        print("❌ No consistency trials found. Run consistency_study.py first.")
        return

    latest_file = trial_files[-1]
    print(f"Loading trials from: {latest_file.name}")

    with open(latest_file) as f:
        data = json.load(f)

    # Load any existing scores from previous runs (resume support)
    existing_scores = load_existing_scores(consistency_dir)
    print(f"Found {len(existing_scores)} previously scored trials")

    # Load DAGs
    dags = load_all_dags()
    dag_by_id = {d["id"]: d for d in dags}

    # Initialize judge
    print(f"Initializing judge: {JUDGE_MODEL}")
    judge = get_client(JUDGE_MODEL)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    scored_trials = {
        "timestamp": timestamp,
        "judge": JUDGE_MODEL,
        "source_file": latest_file.name,
        "config": data["config"],
        "scored_trials": []
    }

    n_new = 0
    n_skipped = 0
    n_failed = 0

    print(f"\nScoring {len(data['trials'])} trials...\n")

    for i, trial in enumerate(data["trials"], 1):
        if trial.get("error") or not trial.get("response"):
            continue

        key = (trial["llm"], trial["dag_id"], trial["task_type"], trial["trial"])

        # Resume: skip if already scored
        if key in existing_scores:
            scored_trials["scored_trials"].append(existing_scores[key])
            existing = existing_scores[key]
            avg = sum(existing["scores"].values()) / len(existing["scores"])
            print(f"  [{i}/{len(data['trials'])}] {trial['llm']} | {trial['task_type']} | {trial['dag_id']} (trial {trial['trial']}) ⏭ already scored (avg={avg:.2f})")
            n_skipped += 1
            continue

        dag = dag_by_id[trial["dag_id"]]
        print(f"  [{i}/{len(data['trials'])}] {trial['llm']} | {trial['task_type']} | {trial['dag_id']} (trial {trial['trial']})...", end=" ", flush=True)

        try:
            prompt = build_judge_prompt(dag, trial["task_type"], trial["response"])
            parsed = score_with_retry(judge, prompt)

            clean_scores = {}
            for metric, val in parsed.get("scores", {}).items():
                try:
                    clean_scores[metric] = int(val)
                except (TypeError, ValueError):
                    continue

            scored_trials["scored_trials"].append({
                "llm": trial["llm"],
                "dag_id": trial["dag_id"],
                "task_type": trial["task_type"],
                "trial": trial["trial"],
                "scores": clean_scores,
                "justification": parsed.get("justification", ""),
            })

            if clean_scores:
                avg = sum(clean_scores.values()) / len(clean_scores)
                print(f"✓ avg={avg:.2f}")
                n_new += 1
            else:
                print(f"❌ no valid scores")
                n_failed += 1

        except Exception as e:
            print(f"❌ {str(e)[:80]}")
            scored_trials["scored_trials"].append({
                "llm": trial["llm"],
                "dag_id": trial["dag_id"],
                "task_type": trial["task_type"],
                "trial": trial["trial"],
                "scores": None,
                "error": str(e)[:200],
            })
            n_failed += 1

        time.sleep(0.3)

        # Save incrementally every 20 entries to avoid losing data on crash
        if (n_new + n_failed) % 20 == 0 and (n_new + n_failed) > 0:
            output_file = consistency_dir / f"consistency_scores_{timestamp}.json"
            with open(output_file, "w") as f:
                json.dump(scored_trials, f, indent=2)

    # Final save
    output_file = consistency_dir / f"consistency_scores_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(scored_trials, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"✅ Scoring complete")
    print(f"  Newly scored: {n_new}")
    print(f"  Skipped (already done): {n_skipped}")
    print(f"  Failed: {n_failed}")
    print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    score_consistency()