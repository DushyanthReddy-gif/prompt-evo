from __future__ import annotations

from typing import Any

from prompt_evo.operators.base import SlotOperator


class FewShotOperator(SlotOperator):
    """Operator 4: add/remove/replace/rewrite few-shot examples (drawn from the dev pool where possible)."""
    name, slot = "few_shot", "few_shot_examples"

    def extra_context(self, parent, ctx) -> str:
        used = {e.example_id for e in parent.few_shot_examples}
        cands = [e for e in ctx.pool if e.id not in used]
        cands = ctx.rng.sample(cands, min(12, len(cands)))
        lines = "\n".join(f"[{e.id}] {e.input} => {e.expected_output}" for e in cands)
        return f"CANDIDATE_EXAMPLES (reference one as {{\"id\": \"<id>\"}}):\n{lines}\n"

    def coerce(self, value: Any, parent, ctx):
        if not isinstance(value, list):
            raise ValueError("few_shot_examples must be a list")
        pool = {e.id: e for e in ctx.pool}
        out: list[dict] = []
        for item in value[: ctx.max_fewshot]:
            if not isinstance(item, dict):
                raise ValueError("example must be an object")
            if "id" in item and "input" not in item:
                if item["id"] not in pool:
                    raise ValueError(f"unknown example id {item['id']}")
                e = pool[item["id"]]
                out.append({"input": e.input, "output": e.expected_output, "example_id": e.id})
            elif "input" in item and "output" in item:
                out.append({"input": str(item["input"]), "output": str(item["output"]), "example_id": item.get("example_id")})
            else:
                raise ValueError("example needs 'id' or 'input'+'output'")
        return out
