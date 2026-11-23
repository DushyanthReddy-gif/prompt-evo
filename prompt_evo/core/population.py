"""Population container with duplicate tracking."""
from __future__ import annotations

from prompt_evo.core.individual import Individual


class Population:
    def __init__(self) -> None:
        self.individuals: list[Individual] = []
        self.seen: set[str] = set()   # hashes of every prompt ever generated (incl. discarded)
        self._counter = 0

    def next_id(self) -> str:
        pid = f"p{self._counter}"
        self._counter += 1
        return pid

    def is_duplicate(self, h: str) -> bool:
        return h in self.seen

    def register(self, h: str) -> None:
        self.seen.add(h)

    def add(self, ind: Individual) -> None:
        self.individuals.append(ind)

    def __len__(self) -> int:
        return len(self.individuals)

    def mean_accuracy(self) -> float:
        return sum(i.accuracy for i in self.individuals) / len(self.individuals) if self.individuals else 0.0
