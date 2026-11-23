from __future__ import annotations

from typing import Any

from prompt_evo.operators.base import SlotOperator


class InstructionWordingOperator(SlotOperator):
    """Operator 1: rewrite instruction wording, preserving the task."""
    name, slot = "instruction_wording", "instruction_wording"

    def coerce(self, value: Any, parent, ctx):
        if isinstance(value, str):
            value = [ln for ln in value.splitlines() if ln.strip()]
        if not isinstance(value, list) or not 1 <= len(value) <= 12 or not all(isinstance(s, str) and s.strip() for s in value):
            raise ValueError("instruction_wording must be 1-12 non-empty strings")
        if any(len(s) > 500 for s in value):
            raise ValueError("instruction too long")
        return value

    def extra_updates(self, parent, value):
        # keep the ordering only if the instruction count is unchanged
        return {} if len(value) == len(parent.instruction_wording) else {"instruction_ordering": []}
