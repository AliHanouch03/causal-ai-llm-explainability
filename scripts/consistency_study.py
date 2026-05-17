"""
Consistency study: re-run a representative subset multiple times to measure
LLM output variance across trials.

Design:
  - 3 DAGs (one per difficulty level: L1, L2, L3)
  - 6 LLMs (all evaluated models)
  - 3 task types (explanation, counterfactual, critique)
  - 3 trials each

Total: 3 DAGs x 6 LLMs x 3 tasks x 3 trials = 162 evaluations
"""

import json
import time
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv(Path(__file__).parent.parent / "src" / ".env")

from dag_loader import load_all_dags
from llm_clients import list_llms, get_client
from tasks import run_task


SELECTED_DAG_IDS = [
    "L1_smoking_cancer",
    "L2_education_income_health",
    "L3_asia_correct",
]
TASK_TYPES = ["explanation", "counterfactual", "critique"]
N_TRIALS = 3


def load_existing_trials(consistency_dir):
    """
    Load all trials from all previous consistency runs.
    Returns a dict keyed by (llm, dag_id, task_type, trial) -> trial record.
    Only successful trials are loaded.
    """
    existing = {}
    for filepath in consistency_dir.glob("consistency_trials_*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)
            for trial in data.get("trials", []):
                if trial.get("error") or not trial.get("response"):
                    continue
                key = (
                    trial["llm"],
                    trial["dag_id"],
                    trial["task_type"],
                    trial["trial"],
                )
                existing[key] = trial
        except Exception:
            continue
    return existing


def run_with_retry(client, dag, task_type, max_retries=4):
    last_error = None
    for attempt in range(max_retries):
        try:
            return run_task(client, dag, task_type)
        except Exception as e:
            last_error = e
            error_str = str(e)
            is_rate_limit = (
                "429" in error_str
                or "rate_limit" in error_str.lower()
                or "RESOURCE_EXHAUSTED" in error_str
                or "quota" in error_str.lower()
            )
            is_timeout = "timeout" in error_str.lower() or "timed out" in error_str.lower()
            if is_rate_limit:
                retry_match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_str)
                wait_time = float(retry_match.group(1)) + 3 if retry_match else 30
                print(f"⏸ rate limited, waiting {wait_time:.0f}s ", end="", flush=True)
                time.sleep(wait_time)
                continue
            elif is_timeout:
                print(f"⏸ timeout, waiting 10s ", end="", flush=True)
                time.sleep(10)
                continue
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries: {last_error}")


def run_consistency_study():
    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / "consistency"
    output_path.mkdir(parents=True, exist_ok=True)

    # Load DAGs
    all_dags = load_all_dags()
    dag_by_id = {d["id"]: d for d in all_dags}
    selected_dags = [dag_by_id[did] for did in SELECTED_DAG_IDS]
    llms = list_llms()

    # Load existing successful trials from any previous run
    existing_trials = load_existing_trials(output_path)
    print(f"Found {len(existing_trials)} existing successful trials from previous runs")

    print(f"\nConsistency study configuration:")
    print(f"  - DAGs: {SELECTED_DAG_IDS}")
    print(f"  - LLMs: {llms}")
    print(f"  - Tasks: {TASK_TYPES}")
    print(f"  - Trials per condition: {N_TRIALS}")
    total = len(selected_dags) * len(llms) * len(TASK_TYPES) * N_TRIALS
    print(f"  - Total target evaluations: {total}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = {
        "timestamp": timestamp,
        "config": {
            "dags": SELECTED_DAG_IDS,
            "llms": llms,
            "tasks": TASK_TYPES,
            "n_trials": N_TRIALS,
        },
        "trials": []
    }

    n_skipped = 0
    n_new = 0
    n_failed = 0

    for llm_name in llms:
        print(f"\n=== {llm_name} ===")
        try:
            client = get_client(llm_name)
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            continue

        for dag in selected_dags:
            for task_type in TASK_TYPES:
                if task_type == "counterfactual" and dag["id"].startswith("L3_asia_broken"):
                    continue

                for trial in range(1, N_TRIALS + 1):
                    key = (llm_name, dag["id"], task_type, trial)
                    
                    # Resume: skip if this exact trial was already done successfully
                    if key in existing_trials:
                        all_results["trials"].append(existing_trials[key])
                        print(f"  [{task_type}] {dag['id']} (trial {trial}/{N_TRIALS}) ⏭ already done")
                        n_skipped += 1
                        continue
                    
                    print(f"  [{task_type}] {dag['id']} (trial {trial}/{N_TRIALS})...", end=" ", flush=True)
                    
                    try:
                        start = time.time()
                        response = run_with_retry(client, dag, task_type)
                        elapsed = time.time() - start
                        
                        all_results["trials"].append({
                            "llm": llm_name,
                            "dag_id": dag["id"],
                            "task_type": task_type,
                            "trial": trial,
                            "response": response,
                            "elapsed_seconds": round(elapsed, 2),
                            "error": None,
                        })
                        print(f"✓ ({elapsed:.1f}s)")
                        n_new += 1
                        
                    except Exception as e:
                        all_results["trials"].append({
                            "llm": llm_name,
                            "dag_id": dag["id"],
                            "task_type": task_type,
                            "trial": trial,
                            "response": None,
                            "elapsed_seconds": None,
                            "error": str(e)[:200],
                        })
                        print(f"❌ {str(e)[:60]}")
                        n_failed += 1
                    
                    time.sleep(0.3)
        
        # Save incrementally after each LLM finishes
        output_file = output_path / f"consistency_trials_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"✅ Consistency study complete.")
    print(f"   Newly run: {n_new}")
    print(f"   Skipped (already done): {n_skipped}")
    print(f"   Failed: {n_failed}")
    print(f"   Total trials: {len(all_results['trials'])}")
    print(f"   Saved to: {output_file}")


if __name__ == "__main__":
    run_consistency_study()