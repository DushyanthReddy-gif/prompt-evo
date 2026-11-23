"""Templates for crossover, feedback and the generic (baseline) operators."""
from __future__ import annotations

SLOT_NAMES = "instruction_wording (list of str), instruction_ordering (permutation of indices), constraints (list of str), few_shot_examples (list of {input, output}), output_format (str, must contain 'answer'), reasoning_instruction (str)"

CROSSOVER = """OPERATOR: crossover
TASK DESCRIPTION: {task}
PARENT_A_JSON: {a}
PARENT_B_JSON: {b}
OBJECTIVE: Combine the most useful components of Parent A and Parent B into one coherent child prompt (e.g. wording from one, constraints or output format from the other). The child must remain consistent and keep the task unchanged.
RULES: Return all six slots: {slots}. instruction_ordering must be a permutation matching the length of instruction_wording.
Respond with ONLY a JSON object: {{"child": {{<six slots>}}, "reason": "<one sentence>"}}"""

FEEDBACK = """OPERATOR: feedback
TASK DESCRIPTION: {task}
PARENT_JSON: {a}
FAILURES (input / expected / model output):
{failures}
OBJECTIVE: Revise the prompt to fix the observed failures without breaking the task or the output format.
RULES: Return all six slots: {slots}. Make targeted edits only.
Respond with ONLY a JSON object: {{"child": {{<six slots>}}, "reason": "<one sentence>"}}"""

GENERIC_MUTATION = """OPERATOR: generic_mutation
TASK DESCRIPTION: {task}
PROMPT:
<<<
{prompt}
>>>
OBJECTIVE: Produce an improved variant of the prompt for the task. Keep a final answer line of the form 'Answer: <answer>'.
Respond with ONLY a JSON object: {{"prompt": "<full new prompt text>", "reason": "<one sentence>"}}"""

GENERIC_CROSSOVER = """OPERATOR: generic_crossover
TASK DESCRIPTION: {task}
PROMPT_A:
<<<
{a}
>>>
PROMPT_B:
<<<
{b}
>>>
OBJECTIVE: Cross over the two prompts: combine their strengths into one new prompt. Keep a final answer line of the form 'Answer: <answer>'.
Respond with ONLY a JSON object: {{"prompt": "<full new prompt text>", "reason": "<one sentence>"}}"""
