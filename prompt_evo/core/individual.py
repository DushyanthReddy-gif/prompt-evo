"""Population member: a fully (dev-set) evaluated prompt."""
from __future__ import annotations

from pydantic import BaseModel, Field

from prompt_evo.evaluation.bootstrap import BootstrapResult
from prompt_evo.schemas.evaluation import EvaluationResult
from prompt_evo.schemas.prompt import StructuredPrompt
from prompt_evo.utils import now_iso


class Individual(BaseModel):
    id: str
    prompt: StructuredPrompt
    text: str
    hash: str
    accuracy: float              # raw accuracy on the full dev set (never hidden)
    fitness: float               # accuracy - lambda * normalized_length
    length: int
    generation: int
    parent_ids: list[str] = Field(default_factory=list)
    operator: str = "seed"
    evaluation: EvaluationResult
    bootstrap: BootstrapResult | None = None
    created_at: str = Field(default_factory=now_iso)

    def summary(self) -> dict:
        d = self.model_dump(exclude={"evaluation", "prompt"})
        d["bootstrap_mean"] = self.bootstrap.mean if self.bootstrap else None
        d["bootstrap_std"] = self.bootstrap.std if self.bootstrap else None
        d["confidence_interval"] = [self.bootstrap.ci_low, self.bootstrap.ci_high] if self.bootstrap else None
        d["raw_accuracy"], d["penalized_fitness"] = self.accuracy, self.fitness
        return d
