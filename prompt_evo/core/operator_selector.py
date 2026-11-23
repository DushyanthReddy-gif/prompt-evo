"""Operator choice: fixed probabilities (default, keeps ablations valid) or Thompson sampling (optional)."""
from __future__ import annotations

import random
from typing import Protocol


class OperatorSelector(Protocol):
    def choose(self, rng: random.Random) -> str: ...
    def update(self, name: str, reward: float) -> None: ...
    def state(self) -> dict: ...


class FixedSelector:
    def __init__(self, names: list[str], weights: dict[str, float] | None = None) -> None:
        self.names = list(names)
        self.weights = [float((weights or {}).get(n, 1.0)) for n in self.names]

    def choose(self, rng: random.Random) -> str:
        return rng.choices(self.names, weights=self.weights, k=1)[0]

    def update(self, name: str, reward: float) -> None:  # fixed: ignores feedback
        return None

    def state(self) -> dict:
        return {"type": "fixed", "weights": dict(zip(self.names, self.weights))}


class ThompsonSelector:
    """Beta-Bernoulli Thompson sampling; reward 1 if the child improved on its parent else 0."""

    def __init__(self, names: list[str]) -> None:
        self.names = list(names)
        self.alpha = {n: 1.0 for n in names}
        self.beta = {n: 1.0 for n in names}

    def choose(self, rng: random.Random) -> str:
        draws = {n: rng.betavariate(self.alpha[n], self.beta[n]) for n in self.names}
        return max(self.names, key=lambda n: draws[n])

    def update(self, name: str, reward: float) -> None:
        if name in self.alpha:
            self.alpha[name] += reward
            self.beta[name] += 1.0 - reward

    def state(self) -> dict:
        return {"type": "thompson", "alpha": dict(self.alpha), "beta": dict(self.beta)}
