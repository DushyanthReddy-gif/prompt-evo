"""Bootstrap robustness estimates (resampling cached per-example outcomes: no extra worker calls)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from pydantic import BaseModel


class BootstrapResult(BaseModel):
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n_samples: int


def bootstrap_accuracy(correct: Sequence[bool | int], n_samples: int = 100, seed: int = 0, level: float = 0.95) -> BootstrapResult:
    x = np.asarray(correct, dtype=float)
    if x.size == 0:
        return BootstrapResult(mean=0.0, std=0.0, ci_low=0.0, ci_high=0.0, n_samples=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_samples, x.size))
    means = x[idx].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return BootstrapResult(mean=float(means.mean()), std=float(means.std(ddof=1)) if n_samples > 1 else 0.0,
                           ci_low=float(lo), ci_high=float(hi), n_samples=n_samples)


def bootstrap_mean_ci(values: Sequence[float], n_samples: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_samples, x.size))].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return float(lo), float(hi)


def paired_bootstrap_delta(a: Sequence[bool | int], b: Sequence[bool | int], n_samples: int = 1000, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap of mean(a)-mean(b) over aligned examples; includes P(delta>0)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_samples, d.size))].mean(axis=1)
    return {"mean": float(d.mean()), "ci_low": float(np.quantile(means, 0.025)),
            "ci_high": float(np.quantile(means, 0.975)), "p_positive": float((means > 0).mean())}
