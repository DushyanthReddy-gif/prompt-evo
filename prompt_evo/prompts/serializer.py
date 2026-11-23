"""StructuredPrompt -> executable prompt text."""
from __future__ import annotations

from prompt_evo.schemas.prompt import StructuredPrompt
from prompt_evo.utils import normalize_ws, sha1_text


def serialize_prompt(p: StructuredPrompt) -> str:
    """Render a structured prompt as the plain-text system prompt sent to the worker."""
    parts = ["\n".join(p.instruction_wording[i] for i in p.instruction_ordering)]
    if p.constraints:
        parts.append("Constraints:\n" + "\n".join(f"- {c}" for c in p.constraints))
    if p.reasoning_instruction:
        parts.append("Reasoning: " + p.reasoning_instruction)
    if p.output_format:
        parts.append("Output format: " + p.output_format)
    if p.few_shot_examples:
        parts.append("Examples:\n" + "\n\n".join(f"Input: {e.input}\nOutput: {e.output}" for e in p.few_shot_examples))
    return "\n\n".join(parts)


def prompt_hash(p: StructuredPrompt) -> str:
    """Duplicate-detection key: whitespace/case-normalized rendered prompt."""
    return sha1_text(normalize_ws(serialize_prompt(p)))
