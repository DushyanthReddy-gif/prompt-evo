"""Candidate racing: stop evaluating hopeless candidates early."""
from __future__ import annotations

import math

from pydantic import BaseModel


class RacingPolicy(BaseModel):
    enabled: bool = True
    min_examples: int = 10
    margin: float = 0.0
    use_bound: bool = True
    use_confidence: bool = False
    z: float = 1.96


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + spread) / denom)


def should_stop(policy: RacingPolicy, n_done: int, n_correct: int, n_total: int, reference: float) -> bool:
    """True if the candidate should stop being evaluated.

    bound rule:      (n_correct + remaining) / n_total < reference - margin  (cannot reach reference)
    confidence rule: Wilson upper bound of current accuracy < reference - margin (after min_examples)
    """
    if not policy.enabled or n_done >= n_total:
        return False
    if policy.use_bound and (n_correct + (n_total - n_done)) / n_total < reference - policy.margin - 1e-12:
        return True
    if policy.use_confidence and n_done >= policy.min_examples:
        return wilson_upper(n_correct, n_done, policy.z) < reference - policy.margin
    return False
