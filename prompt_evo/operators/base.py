"""Operator base classes. Every operator makes an LLM call, validates the structured reply and
returns a NEW prompt object (the parent is never mutated)."""
from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from prompt_evo.core.budget import BudgetExhausted, BudgetManager
from prompt_evo.llm.base import LLMError, OptimizerModel
from prompt_evo.prompts.mutation_templates import OBJECTIVES, REQUEST, RULES, SLOT_OF, SYSTEM_PROMPT
from prompt_evo.prompts.parser import check_compatible
from prompt_evo.prompts.serializer import serialize_prompt
from prompt_evo.schemas.dataset import Example
from prompt_evo.schemas.evolution import MutationRecord
from prompt_evo.schemas.prompt import StructuredPrompt
from prompt_evo.utils import extract_json


@dataclass
class OperatorContext:
    task_description: str
    optimizer: OptimizerModel
    budget: BudgetManager
    rng: random.Random
    pool: list[Example] = field(default_factory=list)       # candidate few-shot examples (dev data)
    failures: list[dict[str, str]] = field(default_factory=list)  # for the feedback operator
    max_retries: int = 1
    max_fewshot: int = 6


@dataclass
class MutationOutcome:
    child: StructuredPrompt
    record: MutationRecord


class InvalidMutation(Exception):
    def __init__(self, message: str, call_ids: list[str] | None = None) -> None:
        super().__init__(message)
        self.call_ids = call_ids or []


def jsonable(v: Any) -> Any:
    return json.loads(json.dumps(v, default=lambda o: o.model_dump()))


class Operator(ABC):
    name: str = "operator"
    slot: str = ""
    n_parents: int = 1

    @abstractmethod
    def build_user_prompt(self, parents: list[StructuredPrompt], ctx: OperatorContext) -> str: ...

    @abstractmethod
    def parse_response(self, data: dict, parents: list[StructuredPrompt], ctx: OperatorContext
                       ) -> tuple[StructuredPrompt, Any, Any, str, dict[str, str] | None]:
        """Return (child, original_component, new_component, reason, slot_sources). Raise ValueError if invalid."""

    def apply(self, parents: list[StructuredPrompt], ctx: OperatorContext) -> MutationOutcome:
        base_user = self.build_user_prompt(parents, ctx)
        errors: list[str] = []
        ids: list[str] = []
        calls = 0
        for _ in range(ctx.max_retries + 1):
            user = base_user if not errors else (
                base_user + f"\n\nYour previous reply was invalid ({errors[-1][:200]}). Reply with valid JSON only.")
            try:
                ctx.budget.reserve_optimizer()
            except BudgetExhausted:
                errors.append("optimizer budget exhausted")
                break
            calls += 1
            try:
                resp = ctx.optimizer.complete(SYSTEM_PROMPT, user)
            except LLMError as e:
                errors.append(f"llm error: {e}")
                continue
            ctx.budget.add_tokens("optimizer", resp.prompt_tokens, resp.completion_tokens)
            if resp.call_id:
                ids.append(resp.call_id)
            if not (resp.text or "").strip():
                errors.append("empty response")
                continue
            try:
                child, before, after, reason, sources = self.parse_response(extract_json(resp.text), parents, ctx)
                check_compatible(child)
            except (ValueError, KeyError, TypeError) as e:  # pydantic ValidationError is a ValueError
                errors.append(f"invalid output: {e}")
                continue
            rec = MutationRecord(operator=self.name, parent_ids=[p.version_id for p in parents],
                                 original_component=before, new_component=after, reason=reason,
                                 llm_call_ids=ids, optimizer_calls=calls, slot_sources=sources)
            return MutationOutcome(child=child, record=rec)
        raise InvalidMutation("; ".join(errors) or "unknown failure", ids)


class SlotOperator(Operator):
    """Edits exactly one slot of a single parent."""
    objective_key: str = ""

    def extra_context(self, parent: StructuredPrompt, ctx: OperatorContext) -> str:
        return ""

    def coerce(self, value: Any, parent: StructuredPrompt, ctx: OperatorContext) -> Any:
        raise NotImplementedError  # pragma: no cover - abstract by convention

    def extra_updates(self, parent: StructuredPrompt, value: Any) -> dict[str, Any]:
        return {}

    def build_user_prompt(self, parents: list[StructuredPrompt], ctx: OperatorContext) -> str:
        p = parents[0]
        return REQUEST.format(
            operator=self.name, task=ctx.task_description, prompt=serialize_prompt(p), slot=self.slot,
            current=json.dumps(jsonable(getattr(p, self.slot))), extra=self.extra_context(p, ctx),
            objective=OBJECTIVES[self.slot], rules=RULES[self.slot].format(max_fewshot=ctx.max_fewshot))

    def parse_response(self, data, parents, ctx):
        parent = parents[0]
        value = self.coerce(data["new_component"], parent, ctx)
        before = jsonable(getattr(parent, self.slot))
        child = StructuredPrompt.model_validate({**parent.model_dump(), self.slot: value, **self.extra_updates(parent, value)})
        after = jsonable(getattr(child, self.slot))
        if after == before:
            raise ValueError("mutation did not change the component")
        return child, before, after, str(data.get("reason", ""))[:500], None


def operator_slot(name: str) -> str:
    return SLOT_OF[name]
