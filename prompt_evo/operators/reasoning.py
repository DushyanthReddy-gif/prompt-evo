from __future__ import annotations

from typing import Any

from prompt_evo.operators.base import SlotOperator


class ReasoningOperator(SlotOperator):
    """Operator 6: change reasoning guidance (never hard-codes chain-of-thought disclosure)."""
    name, slot = "reasoning", "reasoning_instruction"

    def coerce(self, value: Any, parent, ctx):
        if not isinstance(value, str) or len(value) > 600:
            raise ValueError("reasoning_instruction must be a string <=600 chars")
        return value
