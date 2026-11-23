from __future__ import annotations

from typing import Any

from prompt_evo.operators.base import SlotOperator


class InstructionOrderingOperator(SlotOperator):
    """Operator 2: reorder instructions (permutation only; wording untouched)."""
    name, slot = "instruction_ordering", "instruction_ordering"

    def extra_context(self, parent, ctx) -> str:
        lines = "\n".join(f"{i}: {s}" for i, s in enumerate(parent.instruction_wording))
        return f"INSTRUCTIONS (index: text):\n{lines}\n"

    def coerce(self, value: Any, parent, ctx):
        if not isinstance(value, list) or not all(isinstance(i, int) for i in value):
            raise ValueError("instruction_ordering must be a list of ints")
        return value  # permutation validity enforced by StructuredPrompt
