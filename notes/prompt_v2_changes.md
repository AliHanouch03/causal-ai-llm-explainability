# Prompt Refinement: v1 → v2

## Date
06.05.2026

## Reason for refinement
After analyzing v1 outputs, observed that LLMs were:
- Wasting tokens explaining what DAGs and causal models are (irrelevant to target audience)
- Treating reader as causal-AI experts who needed convincing of basics
- Mixing domain-naive and causal-naive framings

## Audience definition refined
- v1: "non-expert manager" (ambiguous — non-expert in what?)
- v2: "experienced professional in the {domain}, not familiar with causal modeling"

## Vocabulary constraints
- v2 explicitly forbids: "node", "edge", "DAG", "directed acyclic graph", "counterfactual"
- v2 mandates: "factors" / "variables" / "links" / "influences"

## Format constraints
- v1: tutorial / explanatory tone
- v2: report tone, no preamble, direct findings

## Cauza alignment
v2 prompts more closely match the actual Cauza product use case: 
domain experts (operations leads, doctors, managers) needing AI-system 
output translated into actionable reports.

## Implications for thesis
Methodology section can frame this as a two-iteration process:
- Iteration 1: generic prompts → revealed prompt-design problem
- Iteration 2: refined prompts → cleaner evaluation
This shows reflective methodology and that prompt design itself is a research finding.