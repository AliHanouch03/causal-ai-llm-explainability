"""
Score all LLM responses using an LLM-as-judge approach.

- Primary judge: Llama 3.3 70B 
- Fallback judge: Gemini 2.5 Flash (used when scoring Llama's own responses to avoid self-bias)
- Includes retry logic for rate limits and resume capability
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

from dag_loader import load_all_dags, dag_to_text
from llm_clients import get_client


# Judge configuration
JUDGE_MODEL = "nemotron-3-super"  # The main judge LLM to use for scoring responses
SELF_JUDGE_FALLBACK = "gemini-2.5-flash"


# Metrics per task type
METRICS = {
    "explanation": [
        "structural_accuracy",
        "pattern_recognition",
        "completeness",
        "clarity",
        "faithfulness",
    ],
    "counterfactual": [
        "direction_correctness",
        "scope_correctness",
        "mechanism_explanation",
        "causal_reasoning",
    ],
    "critique": [
        "flaw_detection",
        "flaw_classification",
        "false_positive_avoidance",
        "explanation_quality",
    ],
}


METRIC_DESCRIPTIONS = {
    "structural_accuracy": "Does the response correctly identify all nodes and edges from the DAG?",
    "pattern_recognition": "Does it correctly recognize the causal pattern (chain, fork, collider, mediation, etc.)?",
    "completeness": "Does it cover the expected explanation points from ground truth?",
    "clarity": "Is the language clear and accessible to a non-expert?",
    "faithfulness": "Does it avoid inventing relationships not in the DAG?",
    "direction_correctness": "Does the predicted change in each variable go in the correct direction?",
    "scope_correctness": "Does it correctly identify which variables are affected and which are NOT?",
    "mechanism_explanation": "Does it explain how the effect propagates through the graph?",
    "causal_reasoning": "Does the reasoning match expected counterfactual logic?",
    "flaw_detection": "Did it correctly identify the flaw (or correctly say there is none on a correct DAG)?",
    "flaw_classification": "Did it correctly categorize the flaw type (reversed/spurious/missing)? On a correct DAG with no flaws, give 5 if it confirmed correctness.",
    "false_positive_avoidance": "On a correct DAG, did it avoid inventing flaws? On a broken DAG, this is N/A — give 5.",
    "explanation_quality": "Did it justify its assessment with sound reasoning?",
}


def build_judge_prompt(dag, task_type, response):
    """Build a prompt asking the judge LLM to score a response."""
    dag_text = dag_to_text(dag)
    ground_truth = json.dumps(dag["ground_truth"], indent=2)
    metrics = METRICS[task_type]

    metrics_block = "\n".join(
        f"- {m}: {METRIC_DESCRIPTIONS[m]}" for m in metrics
    )

    metric_keys_json = ",\n    ".join(f'"{m}": <integer 1-5>' for m in metrics)

    return f"""You are an expert evaluator scoring an LLM's response to a causal reasoning task.

## The DAG

Causal graph (in the {dag['domain']} domain):
{dag_text}

Pattern: {dag['pattern']}
Description: {dag['description']}

## Ground truth (the expected answer)
{ground_truth}

## The task type
{task_type}

## The LLM's response to evaluate
{response}

## Your job
Score the response on each metric below, on a 1-5 integer scale:
- 1 = completely wrong or missing
- 2 = mostly wrong, only minor correct elements
- 3 = partially correct, significant gaps
- 4 = mostly correct, minor issues
- 5 = fully correct and complete

Metrics to score:
{metrics_block}

## Output format
Respond with ONLY a JSON object in this exact format. Do not include any text before or after the JSON. Do not wrap in markdown code fences.

{{
  "scores": {{
    {metric_keys_json}
  }},
  "justification": "<2-3 sentences explaining your scoring rationale>"
}}
"""


def parse_judge_response(text):
    """Extract the JSON object from the judge's response, robustly."""
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()

    # Strip any reasoning tags that survived
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback 1: extract the first {...} block (greedy)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback 2: find the LAST {...} block (in case there's a thinking JSON before the answer)
    matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL))
    for match in reversed(matches):
        try:
            parsed = json.loads(match.group(0))
            if "scores" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # Fallback 3: look for a "scores": {...} substring directly
    scores_match = re.search(r'"scores"\s*:\s*(\{[^{}]*\})', text, re.DOTALL)
    if scores_match:
        try:
            scores_obj = json.loads(scores_match.group(1))
            return {"scores": scores_obj, "justification": ""}
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from judge response: {text[:300]}")


def already_scored(scores_path, llm_name, dag_id, task_type):
    """
    Check if this response was already successfully scored in a previous run.
    Only counts entries with a non-empty scores dict.
    """
    for filepath in scores_path.glob(f"{llm_name}_scores_*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)
            for s in data.get("scores", []):
                scores = s.get("scores")
                if (s.get("dag_id") == dag_id
                    and s.get("task_type") == task_type
                    and scores is not None
                    and isinstance(scores, dict)
                    and len(scores) > 0):
                    return s
        except Exception:
            continue
    return None


def score_with_retry(judge, prompt, max_retries=4):
    """
    Run a judge call with retry on rate limits, timeouts, and parse failures.
    On parse failure, escalates the prompt to demand stricter JSON output.
    """
    last_error = None
    current_prompt = prompt
    
    for attempt in range(max_retries):
        try:
            judge_response = judge.generate(current_prompt)
            return parse_judge_response(judge_response)
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
            is_parse_error = isinstance(e, (json.JSONDecodeError, ValueError))

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
            elif is_parse_error:
                print(f"⏸ parse error, escalating prompt ", end="", flush=True)
                # Escalate prompt strictness on each retry
                current_prompt = escalate_prompt(prompt, attempt)
                time.sleep(2)
                continue
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries: {last_error}")


def escalate_prompt(original_prompt, attempt):
    """
    Rewrite the prompt to be progressively stricter about JSON-only output.
    Used when the judge produces thinking-text instead of valid JSON.
    """
    if attempt == 0:
        # First retry: prepend a strong format reminder
        prefix = (
            "CRITICAL: You must respond with ONLY a valid JSON object. "
            "Do not include any thinking, reasoning, or explanation outside the JSON. "
            "Do not narrate your evaluation. Output the JSON object directly.\n\n"
        )
        return prefix + original_prompt
    
    elif attempt == 1:
        # Second retry: even stricter, with example
        prefix = (
            "OUTPUT FORMAT REQUIREMENT: Your entire response must be a single JSON object "
            "starting with { and ending with }. No text before, no text after. "
            "No <think> tags. No reasoning narration. Just the JSON.\n\n"
            "Example of correct format:\n"
            '{"scores": {"metric_a": 5, "metric_b": 4}, "justification": "brief reason"}\n\n'
            "Now evaluate the following:\n\n"
        )
        return prefix + original_prompt
    
    else:
        # Final retry: minimal prompt asking only for scores, no justification
        # This is a fallback that drops the justification field to reduce length
        prefix = (
            "Output ONLY this JSON, with integer scores 1-5. No other text whatsoever:\n\n"
        )
        return prefix + original_prompt


def find_latest_results(results_path):
    """Find the most recent results file for each LLM."""
    latest = {}
    for filepath in results_path.glob("*.json"):
        # Filename format: {llm_name}_{YYYYMMDD}_{HHMMSS}.json
        parts = filepath.stem.rsplit("_", 2)
        if len(parts) >= 3:
            llm_name = parts[0]
            file_timestamp = "_".join(parts[1:])
            if llm_name not in latest or file_timestamp > latest[llm_name][1]:
                latest[llm_name] = (filepath, file_timestamp)
    return latest


def score_results(results_dir="data/results", scores_dir="data/scores"):
    project_root = Path(__file__).parent.parent
    results_path = project_root / results_dir
    scores_path = project_root / scores_dir
    scores_path.mkdir(parents=True, exist_ok=True)

    # Build a lookup of DAGs by ID
    dags = load_all_dags()
    dag_by_id = {d["id"]: d for d in dags}

    # Initialize the judge
    print(f"Initializing judge: {JUDGE_MODEL}")
    judge = get_client(JUDGE_MODEL)

    # Find latest result files
    latest_files = find_latest_results(results_path)
    print(f"\nFound {len(latest_files)} LLM result files to score")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grand_total_scored = 0
    grand_total_skipped = 0
    grand_total_failed = 0

    for llm_name, (filepath, _) in latest_files.items():
        print(f"\n=== Scoring {llm_name} (judge: {JUDGE_MODEL}) ===")

        with open(filepath) as f:
            llm_data = json.load(f)

        scored_results = {
            "llm": llm_name,
            "judge": JUDGE_MODEL,
            "timestamp": timestamp,
            "scores": []
        }

        for result in llm_data["results"]:
            if result.get("error") or not result.get("response"):
                continue

            dag_id = result["dag_id"]
            task_type = result["task_type"]
            response = result["response"]

            if dag_id not in dag_by_id:
                print(f"  ⚠️  Unknown DAG: {dag_id}")
                continue

            dag = dag_by_id[dag_id]
            print(f"  [{task_type}] {dag_id}...", end=" ", flush=True)

            # Resume support: skip if already successfully scored
            existing = already_scored(scores_path, llm_name, dag_id, task_type)
            if existing:
                scored_results["scores"].append(existing)
                avg = sum(existing["scores"].values()) / len(existing["scores"])
                print(f"⏭  already scored (avg={avg:.2f})")
                grand_total_skipped += 1
                continue

            try:
                prompt = build_judge_prompt(dag, task_type, response)
                parsed = score_with_retry(judge, prompt)

                # Validate that all expected metrics are present
                expected_metrics = set(METRICS[task_type])
                received_metrics = set(parsed.get("scores", {}).keys())
                missing = expected_metrics - received_metrics

                if missing:
                    print(f"⚠️  missing metrics: {missing}", end=" ")

                # Coerce score values to integers
                clean_scores = {}
                for metric, val in parsed.get("scores", {}).items():
                    try:
                        clean_scores[metric] = int(val)
                    except (TypeError, ValueError):
                        continue

                scored_results["scores"].append({
                    "dag_id": dag_id,
                    "task_type": task_type,
                    "scores": clean_scores,
                    "justification": parsed.get("justification", ""),
                    "judge": JUDGE_MODEL,
                })

                if clean_scores:
                    avg = sum(clean_scores.values()) / len(clean_scores)
                    print(f"✓ avg={avg:.2f}")
                    grand_total_scored += 1
                else:
                    print(f"❌ no valid scores")
                    grand_total_failed += 1

            except Exception as e:
                print(f"❌ {str(e)[:80]}")
                scored_results["scores"].append({
                    "dag_id": dag_id,
                    "task_type": task_type,
                    "scores": None,
                    "error": str(e)[:200],
                    "judge": JUDGE_MODEL,
                })
                grand_total_failed += 1

            # Polite delay between requests
            time.sleep(0.3)

        # Save this LLM's scores
        output_file = scores_path / f"{llm_name}_scores_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(scored_results, f, indent=2)
        print(f"  → Saved to {output_file.name}")

    # Final summary
    print(f"\n{'=' * 50}")
    print(f"✅ Scoring complete")
    print(f"  Newly scored: {grand_total_scored}")
    print(f"  Skipped (already done): {grand_total_skipped}")
    print(f"  Failed: {grand_total_failed}")
    print(f"  Output: {scores_path}")


if __name__ == "__main__":
    score_results()