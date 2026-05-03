from dag_loader import dag_to_text


def build_explanation_prompt(dag):
    dag_text = dag_to_text(dag)
    return f"""You are an expert in causal reasoning explaining a causal graph to a non-expert manager.

Here is a causal graph in the {dag['domain']} domain:

{dag_text}

Context: {dag['description']}

Please explain this graph in clear, simple language without technical jargon. Cover:
1. What the main variables represent
2. The key causal pathways and how they connect
3. What this graph tells us about the relationships involved

Use a friendly, structured tone — like a worker explaining the situation to their boss in a meeting.
"""


def build_counterfactual_prompt(dag):
    dag_text = dag_to_text(dag)
    intervention = dag["ground_truth"]["counterfactual"]["intervention"]
    return f"""You are an expert in causal reasoning. Here is a causal graph in the {dag['domain']} domain:

{dag_text}

Now consider this intervention: {intervention}

Explain in simple, non-technical language what would change in this system if this intervention happened. Focus on:
1. Which variables would be affected and how
2. Which variables would NOT be affected and why
3. The chain of consequences flowing through the graph
"""


def build_critique_prompt(dag):
    dag_text = dag_to_text(dag)
    return f"""You are an expert in causal reasoning reviewing a causal graph in the {dag['domain']} domain:

{dag_text}

Critically evaluate this causal graph. For each of the following, give your assessment:
1. Are all the causal directions plausible based on domain knowledge?
2. Are there any spurious or unjustified edges?
3. Are there any important variables (e.g. confounders) that appear to be missing?
4. Are there any other issues with this graph?

If you find any flaws, explain clearly what they are and why they are problematic.
If the graph appears correct, state that clearly with your reasoning.
"""


# Map task types to prompt builders
TASK_BUILDERS = {
    "explanation": build_explanation_prompt,
    "counterfactual": build_counterfactual_prompt,
    "critique": build_critique_prompt,
}


def run_task(client, dag, task_type):
    """Run a single task on a single DAG with a single LLM."""
    if task_type not in TASK_BUILDERS:
        raise ValueError(f"Unknown task type: {task_type}")
    prompt = TASK_BUILDERS[task_type](dag)
    return client.generate(prompt)