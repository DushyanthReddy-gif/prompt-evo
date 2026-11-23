"""Noise-sized, nested evaluation slices with a fixed seed."""
from __future__ import annotations

import random

from prompt_evo.schemas.dataset import Example


def stage_sizes(initial: int, promotion: int, n: int) -> list[int]:
    """Increasing slice sizes strictly below n, e.g. [20, 50]; the full set is appended by make_slices."""
    sizes: list[int] = []
    for s in (initial, promotion):
        if 0 < s < n and (not sizes or s > sizes[-1]):
            sizes.append(s)
    return sizes


def make_slices(dev: list[Example], sizes: list[int], seed: int) -> list[list[Example]]:
    """Nested prefixes of one seeded shuffle, followed by the full dev set (same order)."""
    order = list(dev)
    random.Random(seed).shuffle(order)
    return [order[:s] for s in sizes] + [order]
