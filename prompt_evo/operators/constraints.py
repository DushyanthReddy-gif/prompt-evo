from __future__ import annotations

from typing import Any

from prompt_evo.operators.base import SlotOperator


class ConstraintsOperator(SlotOperator):
    """Operator 3: add/remove/rewrite constraints (no unnecessary restrictions)."""
    name, slot = "constraints", "constraints"

    def coerce(self, value: Any, parent, ctx):
        if not isinstance(value, list) or len(value) > 8 or not all(isinstance(s, str) and s.strip() for s in value):
            raise ValueError("constraints must be <=8 non-empty strings")
        if any(len(s) > 300 for s in value):
            raise ValueError("constraint too long")
        return value
