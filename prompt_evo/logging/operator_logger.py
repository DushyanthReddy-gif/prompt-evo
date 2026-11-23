"""Per-child operator log and per-operator aggregate statistics (core contribution)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from prompt_evo.evaluation.bootstrap import bootstrap_mean_ci
from prompt_evo.evaluation.racing import wilson_upper
from prompt_evo.schemas.evolution import OperatorLogEntry

WIN_EPS = 1e-9


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p, denom = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - spread)), float(min(1.0, centre + spread))


class OperatorLogger:
    def __init__(self) -> None:
        self.entries: list[OperatorLogEntry] = []

    def add(self, entry: OperatorLogEntry) -> None:
        self.entries.append(entry)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([e.model_dump() for e in self.entries])

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "operator_log.jsonl").open("w") as f:
            for e in self.entries:
                f.write(e.model_dump_json() + "\n")
        df = self.to_dataframe()
        if not df.empty:
            df.assign(examples_fixed=df.examples_fixed.map(len), examples_broken=df.examples_broken.map(len)
                      ).drop(columns=["component_before", "component_after"]).to_csv(d / "operator_log.csv", index=False)

    def stats(self, only_full: bool = False) -> list[dict]:
        return compute_operator_stats(self.entries, only_full=only_full)


def load_operator_log(path: str | Path) -> list[OperatorLogEntry]:
    return [OperatorLogEntry.model_validate_json(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compute_operator_stats(entries: Iterable[OperatorLogEntry], only_full: bool = False, n_boot: int = 1000, seed: int = 0) -> list[dict]:
    """Per-operator statistics, ranked by mean score delta.

    attempts        every generation attempt (incl. invalid / duplicate)
    evaluated       attempts with a paired score delta (partial slices count unless only_full)
    n_wins          evaluated children with score_delta > 0 ("successful mutations")
    win_rate        n_wins / evaluated
    success_rate    evaluated / attempts (operational: produced a valid, non-duplicate, evaluable child)
    accept_rate     children admitted into the population / attempts
    """
    by_op: dict[str, list[OperatorLogEntry]] = {}
    for e in entries:
        by_op.setdefault(e.operator, []).append(e)
    rows = []
    for op, es in by_op.items():
        ev = [e for e in es if e.score_delta is not None and (e.full_eval or not only_full)]
        deltas = np.array([e.score_delta for e in ev], dtype=float)
        acc_d = np.array([e.accuracy_delta for e in ev], dtype=float)
        wins = int((deltas > WIN_EPS).sum()) if len(ev) else 0
        lo, hi = bootstrap_mean_ci(deltas, n_boot, seed) if len(ev) else (float("nan"), float("nan"))
        wlo, whi = wilson_interval(wins, len(ev))
        rows.append({
            "operator": op, "attempts": len(es), "evaluated": len(ev), "n_wins": wins,
            "win_rate": wins / len(ev) if ev else 0.0, "success_rate": len(ev) / len(es),
            "accept_rate": sum(e.accepted for e in es) / len(es),
            "mean_score_delta": float(deltas.mean()) if len(ev) else 0.0,
            "median_score_delta": float(np.median(deltas)) if len(ev) else 0.0,
            "mean_accuracy_delta": float(acc_d.mean()) if len(ev) else 0.0,
            "ci_low": lo, "ci_high": hi, "win_rate_ci_low": wlo, "win_rate_ci_high": whi,
            "avg_examples_fixed": float(np.mean([len(e.examples_fixed) for e in ev])) if ev else 0.0,
            "avg_examples_broken": float(np.mean([len(e.examples_broken) for e in ev])) if ev else 0.0,
            "optimizer_calls": sum(e.optimizer_calls for e in es), "worker_calls": sum(e.worker_calls for e in es),
        })
    return sorted(rows, key=lambda r: (-r["mean_score_delta"], -r["win_rate"]))


def format_stats_table(stats: list[dict]) -> str:
    head = f"{'Operator':<24}{'Attempts':>9}{'Evaluated':>10}{'Win Rate':>10}{'Success':>9}{'Mean dScore':>13}{'Mean dAcc':>11}{'95% CI (dScore)':>22}"
    lines = [head, "-" * len(head)]
    for r in stats:
        lines.append(f"{r['operator']:<24}{r['attempts']:>9}{r['evaluated']:>10}{r['win_rate']:>10.0%}{r['success_rate']:>9.0%}"
                     f"{r['mean_score_delta']:>+13.4f}{r['mean_accuracy_delta']:>+11.4f}   [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
    return "\n".join(lines)
