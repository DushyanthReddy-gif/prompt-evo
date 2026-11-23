"""Schemas for mutation records, operator logs and method results."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from prompt_evo.utils import now_iso


class MutationRecord(BaseModel):
    operator: str
    parent_ids: list[str]
    original_component: Any = None
    new_component: Any = None
    reason: str = ""
    llm_call_ids: list[str] = Field(default_factory=list)
    optimizer_calls: int = 0
    slot_sources: dict[str, str] | None = None  # crossover provenance


class OperatorLogEntry(BaseModel):
    """One row per generated child (including failed attempts)."""

    child_id: str
    parent_ids: list[str]
    generation: int
    operator: str
    status: str  # evaluated | not_promoted | racing_stopped | invalid_mutation | duplicate | too_long | worker_failure | budget_exhausted
    before_score: float | None = None
    after_score: float | None = None
    score_delta: float | None = None
    before_accuracy: float | None = None
    after_accuracy: float | None = None
    accuracy_delta: float | None = None
    examples_fixed: list[str] = Field(default_factory=list)
    examples_broken: list[str] = Field(default_factory=list)
    examples_unchanged: int = 0
    n_evaluated: int = 0
    full_eval: bool = False
    accepted: bool = False
    prompt_length_before: int = 0
    prompt_length_after: int = 0
    worker_calls: int = 0
    optimizer_calls: int = 0
    component_before: Any = None
    component_after: Any = None
    change_description: str = ""
    llm_call_ids: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=now_iso)


class MethodResult(BaseModel):
    method: str
    seed: int
    available: bool = True
    note: str = ""
    prompt_text: str = ""
    dev_accuracy: float | None = None
    test_accuracy: float | None = None
    test_correctness: dict[str, bool] = Field(default_factory=dict)
    worker_calls: int = 0
    optimizer_calls: int = 0
    prompt_length: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)
