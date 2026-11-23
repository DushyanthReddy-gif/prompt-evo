"""Experiment B1 (operator isolation) and B2 (leave-one-operator-out)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prompt_evo.config import Config
from prompt_evo.evaluation.bootstrap import paired_bootstrap_delta
from prompt_evo.experiments.runner import run_method, save_json, with_overrides
from prompt_evo.experiments.significance import mean_std_ci, t_interval
from prompt_evo.llm.base import OptimizerModel, WorkerModel
from prompt_evo.schemas.dataset import TaskConfig


def _score(mr, rr, use_test: bool) -> float:
    return mr.test_accuracy if use_test else mr.dev_accuracy


def _wins(rr, op: str) -> int:
    return next((r["n_wins"] for r in rr.operator_stats["all_evaluated"] if r["operator"] == op), 0)


def run_b1(cfg: Config, task: TaskConfig, worker: WorkerModel, opt: OptimizerModel, out_dir: str | Path) -> list[dict]:
    """Seed -> evolve with ONE operator only (no crossover) -> compare gain over the seed."""
    out, use_test = Path(out_dir), bool(task.test_dataset)
    rows = []
    for op in cfg.optimizer.operators:
        c = with_overrides(cfg, optimizer__operators=[op], optimizer__crossover_probability=0.0, optimizer__feedback_probability=0.0)
        gains, dev_gains, wins = [], [], []
        for s in cfg.experiment.seeds:
            mr, rr = run_method("promptevo", c, task, worker, opt, s, run_dir=out / "runs" / f"b1_{op}_seed{s}")
            seed_mr, _ = run_method("seed", c, task, worker, opt, s)
            gains.append(_score(mr, rr, use_test) - _score(seed_mr, None, use_test))
            dev_gains.append(mr.dev_accuracy - rr.seed_individual.accuracy)
            wins.append(_wins(rr, op))
        rows.append({"operator": op, "gain_metric": "test" if use_test else "dev", **{f"gain_{k}": v for k, v in mean_std_ci(gains).items()},
                     "mean_dev_gain": float(np.mean(dev_gains)), "mean_successful_mutations": float(np.mean(wins))})
    rows.sort(key=lambda r: -r["gain_mean"])
    save_json(out / "b1_operator_isolation.json", rows)
    pd.DataFrame(rows).to_csv(out / "b1_operator_isolation.csv", index=False)
    return rows


def run_b2(cfg: Config, task: TaskConfig, worker: WorkerModel, opt: OptimizerModel, out_dir: str | Path) -> list[dict]:
    """Full PromptEvo vs. PromptEvo with one operator removed (same seeds, same budget)."""
    out, use_test, seeds = Path(out_dir), bool(task.test_dataset), cfg.experiment.seeds
    full = {s: run_method("promptevo", cfg, task, worker, opt, s, run_dir=out / "runs" / f"b2_full_seed{s}") for s in seeds}
    full_scores = [_score(*full[s], use_test) for s in seeds]
    rows = []
    for op in cfg.optimizer.operators:
        c = with_overrides(cfg, optimizer__operators=[o for o in cfg.optimizer.operators if o != op])
        loo = {s: run_method("promptevo", c, task, worker, opt, s, run_dir=out / "runs" / f"b2_without_{op}_seed{s}") for s in seeds}
        diffs = [f - _score(*loo[s], use_test) for f, s in zip(full_scores, seeds)]
        row: dict[str, Any] = {"removed_operator": op, "full_mean": float(np.mean(full_scores)),
                               "without_mean": float(np.mean([_score(*loo[s], use_test) for s in seeds])),
                               "operator_contribution": float(np.mean(diffs)), "contribution_ci_low": t_interval(diffs)[0],
                               "contribution_ci_high": t_interval(diffs)[1],
                               "successful_mutations_in_full": float(np.mean([_wins(full[s][1], op) for s in seeds]))}
        if use_test:  # paired bootstrap over test examples (per-example accuracy averaged over seeds)
            ids = [e.id for e in task.test_dataset]
            a = np.mean([[full[s][0].test_correctness[i] for i in ids] for s in seeds], axis=0)
            b = np.mean([[loo[s][0].test_correctness[i] for i in ids] for s in seeds], axis=0)
            pb = paired_bootstrap_delta(a, b)
            row.update({"example_bootstrap_ci_low": pb["ci_low"], "example_bootstrap_ci_high": pb["ci_high"], "p_full_better": pb["p_positive"]})
        rows.append(row)
    rows.sort(key=lambda r: -r["operator_contribution"])
    save_json(out / "b2_leave_one_out.json", rows)
    pd.DataFrame(rows).to_csv(out / "b2_leave_one_out.csv", index=False)
    return rows
