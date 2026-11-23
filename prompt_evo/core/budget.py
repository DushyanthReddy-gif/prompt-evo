"""Global call/cost budget. The WORKER call budget is a hard cap; optimizer calls are tracked separately."""
from __future__ import annotations

import math
from dataclasses import dataclass


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Pricing:
    per_1k_prompt: float = 0.0
    per_1k_completion: float = 0.0


class BudgetManager:
    def __init__(self, worker_budget: int | None, optimizer_budget: int | None = None,
                 worker_pricing: Pricing | None = None, optimizer_pricing: Pricing | None = None) -> None:
        self.worker_budget, self.optimizer_budget = worker_budget, optimizer_budget
        self.pricing = {"worker": worker_pricing or Pricing(), "optimizer": optimizer_pricing or Pricing()}
        self.worker_calls = self.optimizer_calls = 0
        self.tokens = {"worker": [0, 0], "optimizer": [0, 0]}

    def reserve_worker(self) -> None:
        if self.worker_budget is not None and self.worker_calls >= self.worker_budget:
            raise BudgetExhausted("worker call budget exhausted")
        self.worker_calls += 1

    def reserve_optimizer(self) -> None:
        if self.optimizer_budget is not None and self.optimizer_calls >= self.optimizer_budget:
            raise BudgetExhausted("optimizer call budget exhausted")
        self.optimizer_calls += 1

    def add_tokens(self, kind: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.tokens[kind][0] += prompt_tokens
        self.tokens[kind][1] += completion_tokens

    def cost_of(self, kind: str, prompt_tokens: int, completion_tokens: int) -> float:
        p = self.pricing[kind]
        return prompt_tokens / 1000 * p.per_1k_prompt + completion_tokens / 1000 * p.per_1k_completion

    @property
    def worker_remaining(self) -> float:
        return math.inf if self.worker_budget is None else self.worker_budget - self.worker_calls

    @property
    def optimizer_remaining(self) -> float:
        return math.inf if self.optimizer_budget is None else self.optimizer_budget - self.optimizer_calls

    @property
    def total_calls(self) -> int:
        return self.worker_calls + self.optimizer_calls

    @property
    def total_cost(self) -> float:
        return sum(self.cost_of(k, *self.tokens[k]) for k in self.tokens)

    def snapshot(self) -> dict:
        return {"worker_calls": self.worker_calls, "optimizer_calls": self.optimizer_calls,
                "total_calls": self.total_calls,
                "worker_remaining": None if self.worker_budget is None else self.worker_remaining,
                "worker_tokens": list(self.tokens["worker"]), "optimizer_tokens": list(self.tokens["optimizer"]),
                "estimated_cost": self.total_cost}
