from __future__ import annotations

from typing import Any

from prompt_evo.operators.base import SlotOperator


class OutputFormatOperator(SlotOperator):
    """Operator 5: change the required answer format (must stay parseable by the harness)."""
    name, slot = "output_format", "output_format"

    def coerce(self, value: Any, parent, ctx):
        if not isinstance(value, str) or not value.strip() or len(value) > 400:
            raise ValueError("output_format must be a non-empty string <=400 chars")
        if "answer" not in value.lower():
            raise ValueError("output_format must contain 'answer'")
        return value
