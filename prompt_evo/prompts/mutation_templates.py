"""LLM templates for the six primary mutation operators."""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert prompt engineer. You edit prompts for a FROZEN worker LLM; the worker "
    "is never fine-tuned, only its prompt changes. Reply with a single JSON object and nothing else."
)

OBJECTIVES = {
    "instruction_wording": (
        "Rewrite the instruction sentences to be clearer, more precise and more effective while preserving "
        "exactly the same task and requirements. Keep the same number of sentences where possible."),
    "instruction_ordering": (
        "Reorder the instructions (most important/primary requirement first, supporting details after) "
        "without changing what any instruction says. Return a permutation of the instruction indices."),
    "constraints": (
        "Modify the constraints to remove ambiguity, prevent likely errors, or state requirements more clearly. "
        "You may add, remove or rewrite constraints."),
    "few_shot_examples": (
        "Add, remove, replace or rewrite few-shot examples to help the worker. Prefer examples drawn from the "
        "candidate list (reference them as {\"id\": \"...\"}) that are representative and cover likely mistakes."),
    "output_format": (
        "Modify the required answer format (e.g. a labelled final line, explicit fields, concise answer) so "
        "answers are unambiguous and easy to parse."),
    "reasoning_instruction": (
        "Modify the reasoning guidance (e.g. decomposition, verification, self-checking, step ordering). "
        "Tell the worker to reason internally or to give only what the task requires; do not require "
        "disclosing chain-of-thought."),
}

RULES = {
    "instruction_wording": "Do not change the task. Do not add new requirements.",
    "instruction_ordering": "Return every index 0..n-1 exactly once. Do not alter the semantics.",
    "constraints": "Do not introduce unnecessary restrictions. At most 8 short constraints.",
    "few_shot_examples": "At most {max_fewshot} examples. Each needs non-empty input and output.",
    "output_format": "The format string MUST contain the word 'answer' and stay compatible with automatic answer extraction (e.g. a line 'Answer: <answer>').",
    "reasoning_instruction": "At most 2 sentences.",
}

SLOT_OF = {  # operator name -> slot
    "instruction_wording": "instruction_wording", "instruction_ordering": "instruction_ordering",
    "constraints": "constraints", "few_shot": "few_shot_examples",
    "output_format": "output_format", "reasoning": "reasoning_instruction",
}

REQUEST = """OPERATOR: {operator}
TASK DESCRIPTION: {task}
FULL PARENT PROMPT:
<<<
{prompt}
>>>
COMPONENT: {slot}
CURRENT_COMPONENT_JSON: {current}
{extra}OBJECTIVE: {objective}
RULES: {rules}
Respond with ONLY a JSON object: {{"new_component": <same JSON type as CURRENT_COMPONENT_JSON>, "reason": "<one sentence>"}}"""
