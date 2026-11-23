"""Experiment C (tasks x workers x budgets) and the transfer stretch experiment."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from prompt_evo.config import Config
from prompt_evo.experiments.runner import build_task, eval_on_test, run_method, save_json, with_overrides
from prompt_evo.llm.base import OptimizerModel
from prompt_evo.llm.factory import build_worker
from prompt_evo.logging.operator_logger import compute_operator_stats
from prompt_evo.prompts.parser import parse_prompt
from prompt_evo.prompts.serializer import serialize_prompt


def run_c(cfg: Config, opt: OptimizerModel, out_dir: str | Path, base: Path | None = None) -> dict[str, Any]:
    """Does the best mutation type change across tasks, worker models and budgets?"""
    out = Path(out_dir)
    tasks = [(cfg.task.name, build_task(cfg.task, base))] + [(t.name, build_task(t, base)) for t in cfg.experiment.extra_tasks]
    workers = [(cfg.worker.model, cfg.worker)] + [(w.model, w) for w in cfg.experiment.extra_workers]
    rows, op_rows = [], []
    for tname, task in tasks:
        for wname, wsec in workers:
            worker = build_worker(wsec)
            for budget in cfg.experiment.budgets:
                c = with_overrides(cfg, optimizer__worker_budget=budget, worker=wsec)
                entries, accs = [], []
                for s in cfg.experiment.seeds:
                    mr, rr = run_method("promptevo", c, task, worker, opt, s)
                    entries += rr.operator_log.entries
                    accs.append((mr.test_accuracy if task.test_dataset else mr.dev_accuracy, mr.dev_accuracy, rr.seed_individual.accuracy))
                stats = compute_operator_stats(entries)
                for r in stats:
                    op_rows.append({"task": tname, "worker": wname, "budget": budget, **r})
                best = max(stats, key=lambda r: r["mean_score_delta"])["operator"] if stats else None
                rows.append({"task": tname, "worker": wname, "budget": budget, "mean_final_accuracy": sum(a[0] for a in accs) / len(accs),
                             "mean_dev_accuracy": sum(a[1] for a in accs) / len(accs), "mean_seed_dev_accuracy": sum(a[2] for a in accs) / len(accs),
                             "best_operator": best})
    df = pd.DataFrame(rows)
    summary = {"cells": rows, "best_operator_varies_by": {
        axis: bool(df.groupby([c for c in ("task", "worker", "budget") if c != axis])["best_operator"].nunique().gt(1).any())
        for axis in ("task", "worker", "budget") if df[axis].nunique() > 1}}
    save_json(out / "summary.json", summary)
    df.to_csv(out / "robustness_cells.csv", index=False)
    pd.DataFrame(op_rows).to_csv(out / "robustness_operator_stats.csv", index=False)
    from prompt_evo.viz import plot_budget_performance
    plot_budget_performance(rows, out / "plots" / "performance_across_budgets.png")
    return summary


def run_transfer(cfg: Config, opt: OptimizerModel, out_dir: str | Path, base: Path | None = None) -> dict[str, Any]:
    """Evolve on worker A; evaluate seed and evolved prompt on worker B."""
    if cfg.experiment.transfer_worker is None:
        raise ValueError("experiment.transfer_worker must be configured")
    out, task = Path(out_dir), build_task(cfg.task, base)
    wa, wb = build_worker(cfg.worker), build_worker(cfg.experiment.transfer_worker)
    rows = []
    for s in cfg.experiment.seeds:
        mr, rr = run_method("promptevo", cfg, task, wa, opt, s)
        seed_text = serialize_prompt(parse_prompt(task.seed_prompt))
        rec = {"seed": s, "worker_A": cfg.worker.model, "worker_B": cfg.experiment.transfer_worker.model}
        for label, text in (("seed", seed_text), ("evolved", mr.prompt_text)):
            rec[f"{label}_on_A"], _ = eval_on_test(text, task, wa, cfg)
            rec[f"{label}_on_B"], _ = eval_on_test(text, task, wb, cfg)
        rec["transfer_gain_on_B"] = (rec["evolved_on_B"] - rec["seed_on_B"]) if rec["seed_on_B"] is not None else None
        rows.append(rec)
    summary = {"rows": rows}
    save_json(out / "summary.json", summary)
    pd.DataFrame(rows).to_csv(out / "transfer.csv", index=False)
    return summary
