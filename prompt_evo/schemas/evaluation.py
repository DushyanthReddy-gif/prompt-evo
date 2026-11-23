"""Evaluation result schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ExampleResult(BaseModel):
    example_id: str
    input: str
    expected_output: str
    model_output: str
    correct: bool
    category: str | None = None  # None if correct, else empty_response / no_answer_marker / wrong_answer
    latency: float = 0.0


class EvaluationResult(BaseModel):
    prompt_hash: str
    accuracy: float
    n_correct: int
    n_incorrect: int
    n_evaluated: int
    n_requested: int
    complete: bool
    stop_reason: str | None = None  # racing | budget | timeout
    per_example: list[ExampleResult] = Field(default_factory=list)
    latency: float = 0.0
    worker_calls: int = 0  # fresh (non-cached) calls made by this evaluation
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost: float = 0.0

    def correctness(self) -> dict[str, bool]:
        return {r.example_id: r.correct for r in self.per_example}
