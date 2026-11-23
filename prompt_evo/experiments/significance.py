"""Paired significance tests, effect sizes and intervals."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def paired_permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test on paired differences (Monte Carlo, add-one corrected)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.size == 0 or np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm >= abs(d.mean()) - 1e-12) + 1) / (n_perm + 1))


def wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.size == 0 or np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.size < 2 or d.std(ddof=1) == 0:
        return 0.0 if np.allclose(d, 0) else float("inf")
    return float(d.mean() / d.std(ddof=1))


def t_interval(values: Sequence[float], level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(values, float)
    if x.size < 2:
        return (float("nan"), float("nan"))
    h = stats.sem(x) * stats.t.ppf((1 + level) / 2, x.size - 1)
    return float(x.mean() - h), float(x.mean() + h)


def mean_std_ci(values: Sequence[float]) -> dict[str, float]:
    x = np.asarray(values, float)
    lo, hi = t_interval(x)
    return {"mean": float(x.mean()) if x.size else float("nan"), "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "ci_low": lo, "ci_high": hi, "n": int(x.size)}


def compare_paired(a: Sequence[float], b: Sequence[float], seed: int = 0) -> dict[str, float]:
    d = np.asarray(a, float) - np.asarray(b, float)
    return {"mean_diff": float(d.mean()) if d.size else float("nan"), "p_permutation": paired_permutation_test(a, b, seed=seed),
            "p_wilcoxon": wilcoxon_p(a, b), "cohens_dz": cohens_dz(a, b), "n": int(d.size)}
