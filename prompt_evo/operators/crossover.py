from __future__ import annotations

import json

from prompt_evo.operators.base import Operator, OperatorContext
from prompt_evo.prompts.crossover_templates import CROSSOVER, SLOT_NAMES
from prompt_evo.schemas.prompt import SLOTS, StructuredPrompt


def child_from_reply(data: dict, parents: list[StructuredPrompt]) -> tuple[StructuredPrompt, dict[str, str]]:
    """Validate a full-prompt reply and compute per-slot provenance (A / B / new)."""
    slots = data["child"]
    if not isinstance(slots, dict):
        raise ValueError("'child' must be an object")
    child = StructuredPrompt.model_validate({s: slots[s] for s in SLOTS if s in slots})
    cd = child.slots_dict()
    sources = {}
    for s in SLOTS:
        match = [chr(65 + i) for i, p in enumerate(parents) if p.slots_dict()[s] == cd[s]]
        sources[s] = "/".join(match) if match else "new"
    return child, sources


class CrossoverOperator(Operator):
    """LLM crossover of two parents into one coherent child (validated before use)."""
    name, slot, n_parents = "crossover", "all", 2

    def build_user_prompt(self, parents, ctx: OperatorContext) -> str:
        a, b = parents
        return CROSSOVER.format(task=ctx.task_description, a=json.dumps(a.slots_dict()), b=json.dumps(b.slots_dict()), slots=SLOT_NAMES)

    def parse_response(self, data, parents, ctx):
        child, sources = child_from_reply(data, parents)
        cd = child.slots_dict()
        if any(cd == p.slots_dict() for p in parents):
            raise ValueError("child identical to a parent")
        return child, [p.slots_dict() for p in parents], cd, str(data.get("reason", ""))[:500], sources
