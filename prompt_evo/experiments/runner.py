"""Task building, single-run execution, result saving and Experiment A."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prompt_evo.baselines.genetic import run_genetic
from prompt_evo.baselines.manual import run_manual
from prompt_evo.baselines.seed import run_seed
from prompt_evo.config import Config, TaskSection, resolve_text
from prompt_evo.core.budget import BudgetManager
from prompt_evo.core.optimizer import PromptEvo, RunResult
from prompt_evo.evaluation.evaluator import Evaluator
from prompt_evo.evaluation.metrics import METRICS
from prompt_evo.experiments.significance import compare_paired, mean_std_ci
from prompt_evo.llm.base import OptimizerModel, WorkerModel
from prompt_evo.logging.evolution_logger import EvolutionLogger
from prompt_evo.logging.operator_logger import format_stats_table
from prompt_evo.schemas.dataset import TaskConfig, dataset_hash, load_dataset
from prompt_evo.schemas.evolution import MethodResult
from prompt_evo.utils import git_commit, now_iso


def build_task(sec: TaskSection, base: Path | None = None) -> TaskConfig:
    if sec.type not in METRICS:
        raise ValueError(f"unknown task type {sec.type}; valid: {list(METRICS)}")
    seed = resolve_text(sec.seed_prompt, base)
    if not seed:
        raise ValueError("task.seed_prompt is required")
    return TaskConfig(
        name=sec.name, description=sec.description, evaluator=sec.type, dataset=load_dataset(sec.dataset), seed_prompt=seed,
        test_dataset=load_dataset(sec.test_dataset) if sec.test_dataset else [],
        fewshot_pool=load_dataset(sec.fewshot_pool) if sec.fewshot_pool else [], manual_prompt=resolve_text(sec.manual_prompt, base))


def with_overrides(cfg: Config, **kv: Any) -> Config:
    """Deep-copy ``cfg`` and set dotted keys with '__' (e.g. optimizer__worker_budget=500)."""
    new = cfg.model_copy(deep=True)
    for k, v in kv.items():
        sec, _, field = k.partition("__")
        if field:
            setattr(getattr(new, sec), field, v)
        else:  # whole-section replacement, e.g. worker=<LLMSection>
            setattr(new, sec, v)
    return new


def repro_metadata(cfg: Config, task: TaskConfig, extra: dict | None = None) -> dict[str, Any]:
    o = cfg.optimizer
    return {"timestamp": now_iso(), "git_commit": git_commit(), "random_seed": o.seed, "worker_model": cfg.worker.model,
            "optimizer_model": cfg.optimizer_llm.model, "dataset": cfg.task.dataset, "dataset_hash": dataset_hash(task.dataset),
            "test_dataset_hash": dataset_hash(task.test_dataset) if task.test_dataset else None, "worker_budget": o.worker_budget,
            "population_size": o.population_size, "max_generations": o.max_generations,
            "operator_probabilities": {"mutation": o.mutation_probability, "crossover": o.crossover_probability,
                                       "feedback": o.feedback_probability, "operator_weights": o.operator_weights},
            "config": cfg.model_dump(), **(extra or {})}


def eval_on_test(text: str, task: TaskConfig, worker: WorkerModel, cfg: Config) -> tuple[float | None, dict[str, bool]]:
    """Final held-out evaluation. Uses a separate unlimited budget; never used for selection."""
    if not task.test_dataset:
        return None, {}
    ev = Evaluator(worker, METRICS[task.evaluator], BudgetManager(None), cache=cfg.evaluation.cache)
    r = ev.evaluate(text, task.test_dataset)
    return r.accuracy, r.correctness()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def save_run(result: RunResult, out_dir: str | Path, cfg: Config, task: TaskConfig, test_acc: float | None = None,
             plots: bool = True) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    save_json(d / "config.json", repro_metadata(cfg, task))
    if result.evolution:
        result.evolution.save(d / "evolution.jsonl")
    save_json(d / "operator_stats.json", result.operator_stats)
    result.operator_log.save(d)
    pd.DataFrame(result.operator_stats["all_evaluated"]).to_csv(d / "operator_stats.csv", index=False)
    save_json(d / "final_population.json", [i.summary() | {"text": i.text} for i in result.population])
    pd.DataFrame([i.summary() for i in result.population]).drop(columns=["confidence_interval"]).to_csv(d / "final_population.csv", index=False)
    s = result.seed_individual
    save_json(d / "summary.json", {
        "best_prompt": result.best.text, "best_id": result.best.id, "best_dev_accuracy": result.best.accuracy,
        "best_dev_fitness": result.best.fitness, "best_bootstrap": result.best.bootstrap.model_dump() if result.best.bootstrap else None,
        "seed_dev_accuracy": s.accuracy, "seed_prompt": s.text, "test_accuracy": test_acc, "budget": result.budget,
        "selector_state": result.selector_state, "history": result.history, "generated_at": now_iso()})
    save_json(d / "per_example_results.json", {"best": [r.model_dump() for r in result.best.evaluation.per_example],
                                                "seed": [r.model_dump() for r in s.evaluation.per_example]})
    if plots:
        from prompt_evo.viz import make_run_plots
        make_run_plots(d)
    return d


def run_method(method: str, cfg: Config, task: TaskConfig, worker: WorkerModel, opt: OptimizerModel, seed: int,
               run_dir: Path | None = None, plots: bool = False) -> tuple[MethodResult, RunResult | None]:
    cfg = with_overrides(cfg, optimizer__seed=seed)
    rr: RunResult | None = None
    if method == "seed":
        mr = run_seed(cfg, task, worker, seed)
    elif method == "manual":
        mr = run_manual(cfg, task, worker, seed)
    elif method == "genetic":
        mr = run_genetic(cfg, task, worker, opt, seed)
    elif method == "promptevo":
        rr = PromptEvo(cfg, task, worker, opt, EvolutionLogger()).run()
        mr = MethodResult(method="promptevo", seed=seed, prompt_text=rr.best.text, dev_accuracy=rr.best.accuracy,
                          worker_calls=rr.budget["worker_calls"], optimizer_calls=rr.budget["optimizer_calls"], prompt_length=rr.best.length,
                          extra={"seed_dev_accuracy": rr.seed_individual.accuracy, "history": rr.history})
    else:
        raise ValueError(f"unknown method {method}")
    if mr.available:
        mr.test_accuracy, mr.test_correctness = eval_on_test(mr.prompt_text, task, worker, cfg)
    if rr is not None and run_dir is not None:
        save_run(rr, run_dir, cfg, task, mr.test_accuracy, plots=plots)
    return mr, rr


def run_experiment_a(cfg: Config, task: TaskConfig, worker: WorkerModel, opt: OptimizerModel, out_dir: str | Path) -> dict[str, Any]:
    """Experiment A: seed vs manual vs genetic vs PromptEvo over several seeds, paired statistics."""
    out, seeds = Path(out_dir), cfg.experiment.seeds
    results: dict[str, list[MethodResult]] = {}
    for m in cfg.experiment.methods:
        for s in seeds:
            mr, _ = run_method(m, cfg, task, worker, opt, s, run_dir=out / "runs" / f"{m}_seed{s}" if m == "promptevo" else None, plots=True)
            results.setdefault(m, []).append(mr)
    use_test = bool(task.test_dataset)
    metric = (lambda r: r.test_accuracy) if use_test else (lambda r: r.dev_accuracy)
    seed_vals = [metric(r) for r in results.get("seed", [])]
    rows = []
    for m, rs in results.items():
        if not rs[0].available:
            rows.append({"method": m, "available": False, "note": rs[0].note})
            continue
        vals = [metric(r) for r in rs]
        row = {"method": m, "available": True, "metric": "test_accuracy" if use_test else "dev_accuracy (no test set)",
               **mean_std_ci(vals), "mean_dev_accuracy": float(np.mean([r.dev_accuracy for r in rs])),
               "mean_worker_calls": float(np.mean([r.worker_calls for r in rs])),
               "mean_optimizer_calls": float(np.mean([r.optimizer_calls for r in rs])),
               "mean_prompt_length": float(np.mean([r.prompt_length for r in rs]))}
        if seed_vals and m != "seed":
            row["improvement_over_seed"] = float(np.mean(np.array(vals) - np.array(seed_vals)))
        rows.append(row)
    tests = {}
    if use_test:
        def per_ex(m: str) -> np.ndarray:  # per-example accuracy averaged over seeds
            ids = [e.id for e in task.test_dataset]
            return np.mean([[float(r.test_correctness[i]) for i in ids] for r in results[m]], axis=0)
        for a, b in [("promptevo", "seed"), ("promptevo", "manual"), ("promptevo", "genetic"), ("genetic", "seed")]:
            if a in results and b in results and results[a][0].available and results[b][0].available:
                tests[f"{a}_vs_{b}"] = compare_paired(per_ex(a), per_ex(b))
    summary = {"rows": rows, "paired_tests_over_test_examples": tests, "seeds": seeds, "generated_at": now_iso()}
    save_json(out / "summary.json", summary)
    save_json(out / "config.json", repro_metadata(cfg, task, {"seeds": seeds, "methods": cfg.experiment.methods}))
    save_json(out / "per_example_results.json", {m: {str(r.seed): r.test_correctness for r in rs} for m, rs in results.items()})
    save_json(out / "prompts.json", {m: {str(r.seed): r.prompt_text for r in rs} for m, rs in results.items()})
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    from prompt_evo.viz import make_experiment_plots
    make_experiment_plots(out, rows, results)
    return summary


def format_summary(summary: dict) -> str:
    lines = [f"{'method':<12}{'mean':>8}{'std':>8}{'95% CI':>20}{'d vs seed':>11}{'calls':>8}{'opt calls':>10}{'len':>7}"]
    for r in summary["rows"]:
        if not r.get("available"):
            lines.append(f"{r['method']:<12}UNAVAILABLE ({r.get('note')})")
            continue
        lines.append(f"{r['method']:<12}{r['mean']:>8.3f}{r['std']:>8.3f}   [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
                     f"{r.get('improvement_over_seed', 0):>+11.3f}{r['mean_worker_calls']:>8.0f}{r['mean_optimizer_calls']:>10.0f}{r['mean_prompt_length']:>7.0f}")
    return "\n".join(lines)
