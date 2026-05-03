"""
Run all LLMs on all DAGs across all tasks.
Saves outputs to data/results/ as JSON files.
Includes retry logic for rate limits and resume capability.
"""

import json
import re
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv(Path(__file__).parent.parent / "src" / ".env")

from dag_loader import load_all_dags
from llm_clients import list_llms, get_client
from tasks import run_task


TASK_PLAN = {
    "explanation": "all",
    "counterfactual": "correct_only",
    "critique": "all",
}


def select_dags_for_task(dags, task_type):
    """Decide which DAGs to use for which task."""
    plan = TASK_PLAN[task_type]
    if plan == "all":
        return dags
    elif plan == "correct_only":
        return [d for d in dags if not d["id"].startswith("L3_asia_broken")]
    else:
        return dags


def already_completed(output_path, llm_name, dag_id, task_type):
    """Check if this task was already successfully completed in a previous run."""
    for filepath in output_path.glob(f"{llm_name}_*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)
            for r in data.get("results", []):
                if (r["dag_id"] == dag_id
                    and r["task_type"] == task_type
                    and r.get("error") is None
                    and r.get("response")):
                    return r
        except Exception:
            continue
    return None


def run_with_retry(client, dag, task_type, max_retries=4):
    """Run a task with automatic retry on rate limits."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return run_task(client, dag, task_type)
        except Exception as e:
            last_error = e
            error_str = str(e)

            # Detect rate-limit-style errors
            is_rate_limit = (
                "429" in error_str
                or "rate_limit" in error_str.lower()
                or "RESOURCE_EXHAUSTED" in error_str
                or "quota" in error_str.lower()
            )
            
            is_timeout = "timeout" in error_str.lower() or "timed out" in error_str.lower()

            if is_rate_limit:
                # Try to parse retry delay from message
                retry_match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_str)
                wait_time = float(retry_match.group(1)) + 3 if retry_match else 60
                print(f"⏸  rate limited, waiting {wait_time:.0f}s ", end="", flush=True)
                time.sleep(wait_time)
                continue
            elif is_timeout:
                print(f"⏸  timeout, waiting 10s ", end="", flush=True)
                time.sleep(10)
                continue
            else:
                # Non-retryable error
                raise
    raise RuntimeError(f"Failed after {max_retries} retries: {last_error}")


def run_evaluation(output_dir="data/results"):
    project_root = Path(__file__).parent.parent
    output_path = project_root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    dags = load_all_dags()
    llms = list_llms()

    print(f"Running evaluation:")
    print(f"  - {len(dags)} DAGs loaded")
    print(f"  - {len(llms)} LLMs to test")
    print(f"  - {len(TASK_PLAN)} task types")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for llm_name in llms:
        print(f"\n=== {llm_name} ===")

        try:
            client = get_client(llm_name)
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            continue

        llm_results = {
            "llm": llm_name,
            "timestamp": timestamp,
            "results": []
        }

        for task_type in TASK_PLAN.keys():
            selected_dags = select_dags_for_task(dags, task_type)

            for dag in selected_dags:
                print(f"  [{task_type}] {dag['id']}...", end=" ", flush=True)

                # Skip if already completed in a previous run
                existing = already_completed(output_path, llm_name, dag["id"], task_type)
                if existing:
                    llm_results["results"].append(existing)
                    print("⏭  already done")
                    continue

                try:
                    start = time.time()
                    response = run_with_retry(client, dag, task_type)
                    elapsed = time.time() - start

                    llm_results["results"].append({
                        "dag_id": dag["id"],
                        "task_type": task_type,
                        "response": response,
                        "elapsed_seconds": round(elapsed, 2),
                        "error": None,
                    })
                    print(f"✓ ({elapsed:.1f}s)")

                except Exception as e:
                    llm_results["results"].append({
                        "dag_id": dag["id"],
                        "task_type": task_type,
                        "response": None,
                        "elapsed_seconds": None,
                        "error": str(e)[:200],
                    })
                    print(f"❌ {str(e)[:80]}")

                # Small polite delay between requests
                time.sleep(0.5)

        output_file = output_path / f"{llm_name}_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(llm_results, f, indent=2)
        print(f"  → Saved to {output_file.name}")

    print(f"\n✅ Evaluation complete. Results in: {output_path}")


if __name__ == "__main__":
    run_evaluation()