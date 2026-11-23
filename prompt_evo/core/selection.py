"""Tournament and Pareto selection (accuracy, prompt length, robustness)."""
from __future__ import annotations

import random

from prompt_evo.core.individual import Individual


def dominates(a: Individual, b: Individual) -> bool:
    """Objectives: maximize accuracy, minimize length, minimize bootstrap std."""
    va = (a.accuracy, -a.length, -(a.bootstrap.std if a.bootstrap else 0.0))
    vb = (b.accuracy, -b.length, -(b.bootstrap.std if b.bootstrap else 0.0))
    return all(x >= y for x, y in zip(va, vb)) and any(x > y for x, y in zip(va, vb))


def pareto_ranks(inds: list[Individual]) -> dict[str, int]:
    """Rank = number of individuals dominating it (0 = non-dominated front)."""
    return {a.id: sum(dominates(b, a) for b in inds if b is not a) for a in inds}


class Selection:
    """score = fitness - robust_penalty * bootstrap_std, so candidates that are noisy lose ties."""

    def __init__(self, mode: str = "tournament", tournament_size: int = 3, robust_penalty: float = 0.5) -> None:
        if mode not in ("tournament", "pareto"):
            raise ValueError(f"unknown selection mode {mode}")
        self.mode, self.k, self.robust_penalty = mode, tournament_size, robust_penalty

    def score(self, ind: Individual) -> float:
        return ind.fitness - self.robust_penalty * (ind.bootstrap.std if ind.bootstrap else 0.0)

    def pick_parent(self, inds: list[Individual], rng: random.Random) -> Individual:
        contestants = rng.sample(inds, min(self.k, len(inds)))
        if self.mode == "pareto":
            ranks = pareto_ranks(inds)
            return min(contestants, key=lambda i: (ranks[i.id], -self.score(i)))
        return max(contestants, key=self.score)

    def survivors(self, inds: list[Individual], n: int) -> list[Individual]:
        if self.mode == "pareto":
            ranks = pareto_ranks(inds)
            return sorted(inds, key=lambda i: (ranks[i.id], -self.score(i), i.length))[:n]
        return sorted(inds, key=lambda i: (-self.score(i), i.length))[:n]

    def best(self, inds: list[Individual]) -> Individual:
        return max(inds, key=lambda i: (self.score(i), -i.length))
