from dag_loader import dag_to_text


def build_explanation_prompt(dag):
    dag_text = dag_to_text(dag)
    return f"""You are translating the output of a causal AI system into a clear report for a domain expert.

The reader is an experienced professional in the {dag['domain']} domain (e.g. a doctor, manager, or operations lead). They understand the subject matter well, but they are not familiar with technical causal modeling concepts like DAGs, structural causal models, or directed graphs.

Your job is to explain what the causal AI system found, in plain language, as a written report.

The causal AI system identified the following structure:

{dag_text}

Each item on the left side of an arrow influences the item on the right side. Refer to these as "factors" or "variables", and to the arrows as "links" or "influences" — never use technical terms like "node", "edge", "DAG", or "directed acyclic graph".

Write a report that:
1. Starts directly with the findings — do NOT explain what causal models or graphs are
2. Identifies the main factors involved and how they relate to each other
3. Explains the key chains of influence — what leads to what, and through what
4. Highlights anything practically important the reader should take away

Tone: like a colleague summarizing analysis for an experienced peer who needs the bottom line, not a tutorial.
"""


def build_counterfactual_prompt(dag):
    dag_text = dag_to_text(dag)
    intervention = dag["ground_truth"]["counterfactual"]["intervention"]
    return f"""You are translating the output of a causal AI system into a clear report for a domain expert.

The reader is an experienced professional in the {dag['domain']} domain. They understand the subject matter well, but they are not familiar with technical concepts like counterfactuals, structural causal models, or DAGs. Do not explain those concepts.

The causal AI system has identified the following structure of influences:

{dag_text}

Refer to these as "factors" or "variables" and the arrows as "links" or "influences" — never use technical terms like "node", "edge", "DAG", or "counterfactual".

The reader is now asking: what would happen if this intervention took place?

**Intervention: {intervention}**

Write a clear report that explains:
1. Which factors would change as a result, and in which direction
2. Which factors would NOT change, and why
3. The chain of consequences — how the effect ripples through the system
4. Any practical implications worth flagging

Tone: like a colleague briefing a decision-maker. Start directly with the impact analysis — no conceptual preamble.
"""


def build_critique_prompt(dag):
    dag_text = dag_to_text(dag)
    return f"""You are reviewing the output of a causal AI system on behalf of a domain expert.

The reader is an experienced professional in the {dag['domain']} domain. They want to know whether the causal AI's findings are reasonable based on real-world domain knowledge. They do not need a tutorial on what causal models or DAGs are.

The causal AI system claims the following structure of influences:

{dag_text}

Refer to these as "factors" and the arrows as "links" or "influences" — never use technical terms like "node", "edge", or "DAG".

Critically evaluate this output, drawing on what is actually known about the {dag['domain']} domain. Address:

1. Are all the directions of influence plausible? (Could any be backwards?)
2. Are there any links that seem to claim something the domain does not actually support?
3. Are there important factors that should probably be in this analysis but appear to be missing?
4. Anything else that looks off, or anything that looks particularly well-captured?

If you find issues, explain clearly what they are and why they matter.
If the analysis looks solid, say so and explain why.

Tone: like a senior colleague giving an honest second opinion. Start directly with your assessment — no preamble explaining what a causal graph is.
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