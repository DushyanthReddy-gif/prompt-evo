from __future__ import annotations

import json

from prompt_evo.operators.base import Operator, OperatorContext
from prompt_evo.operators.crossover import child_from_reply
from prompt_evo.prompts.crossover_templates import FEEDBACK, SLOT_NAMES


class FeedbackOperator(Operator):
    """OPTIONAL operator (not one of the six primary ones): revise the prompt using observed failures."""
    name, slot = "feedback", "all"

    def build_user_prompt(self, parents, ctx: OperatorContext) -> str:
        if not ctx.failures:
            raise ValueError("feedback operator needs failures in the context")
        fails = "\n".join(json.dumps(f) for f in ctx.failures[:5])
        return FEEDBACK.format(task=ctx.task_description, a=json.dumps(parents[0].slots_dict()), failures=fails, slots=SLOT_NAMES)

    def parse_response(self, data, parents, ctx):
        child, sources = child_from_reply(data, parents)
        if child.slots_dict() == parents[0].slots_dict():
            raise ValueError("child identical to parent")
        return child, parents[0].slots_dict(), child.slots_dict(), str(data.get("reason", ""))[:500], sources
