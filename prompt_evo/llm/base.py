"""Provider-agnostic LLM interfaces. WORKER (frozen) and OPTIMIZER (mutation) are distinct types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMError(RuntimeError):
    """Raised after retries are exhausted or on non-retryable API failures."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_id: str | None = None
    latency: float = 0.0


class WorkerModel(ABC):
    """The FROZEN model being prompted. Never trained or modified."""
    name: str = "worker"
    temperature: float = 0.0

    @abstractmethod
    def generate(self, prompt: str, input: str) -> LLMResponse:
        """Run the worker with ``prompt`` (system role) on ``input`` (user role)."""


class OptimizerModel(ABC):
    """The model that proposes prompt edits (mutation/crossover/feedback)."""
    name: str = "optimizer"

    @abstractmethod
    def complete(self, system: str, user: str) -> LLMResponse:
        """Return the optimizer's reply to a mutation request."""
