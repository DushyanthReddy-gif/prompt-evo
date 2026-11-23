"""Structured prompt representation (six slots)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

SLOTS = [
    "instruction_wording",
    "instruction_ordering",
    "constraints",
    "few_shot_examples",
    "output_format",
    "reasoning_instruction",
]


class FewShotExample(BaseModel):
    input: str
    output: str
    example_id: str | None = None

    @field_validator("input", "output")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = " ".join(str(v).split())  # single line: keeps serialization lossless
        if not v:
            raise ValueError("few-shot fields must be non-empty")
        return v


class StructuredPrompt(BaseModel):
    """A prompt split into six independently mutable slots.

    ``instruction_wording`` holds the instruction sentences (index order is canonical);
    ``instruction_ordering`` is a permutation giving the order they are rendered in.
    """

    instruction_wording: list[str] = Field(min_length=1)
    instruction_ordering: list[int] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)
    output_format: str = ""
    reasoning_instruction: str = ""
    raw_prompt: str | None = None
    version_id: str = "p0"
    parent_ids: list[str] = Field(default_factory=list)
    operator: str = "seed"
    generation: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("instruction_wording", "constraints")
    @classmethod
    def _clean_lines(cls, v: list[str]) -> list[str]:
        out = [" ".join(str(s).split()) for s in v]
        if any(not s for s in out):
            raise ValueError("empty line in list slot")
        return out

    @field_validator("output_format", "reasoning_instruction")
    @classmethod
    def _clean_str(cls, v: str) -> str:
        return " ".join(str(v).split())

    @model_validator(mode="after")
    def _check_ordering(self) -> "StructuredPrompt":
        n = len(self.instruction_wording)
        if not self.instruction_ordering:
            self.instruction_ordering = list(range(n))
        elif sorted(self.instruction_ordering) != list(range(n)):
            raise ValueError(f"instruction_ordering must be a permutation of 0..{n - 1}")
        return self

    def slots_dict(self) -> dict[str, Any]:
        return self.model_dump(include=set(SLOTS))
