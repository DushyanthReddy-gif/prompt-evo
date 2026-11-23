"""PromptEvo command line interface."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from prompt_evo.config import load_config


def _setup(args):
    cfg = load_config(args.config)
    from prompt_evo.experiments.runner import build_task
    from prompt_evo.llm.factory import build_optimizer, build_worker
    base = Path(args.config).parent
    return cfg, build_task(cfg.task, base), build_worker(cfg.worker), build_optimizer(cfg.optimizer_llm), base


def _out_dir(cfg, args, name: str | None = None) -> Path:
    return Path(args.output or cfg.experiment.output_dir) / (name or cfg.experiment.name)


def cmd_optimize(args) -> int:
    from prompt_evo.core.optimizer import PromptEvo
    from prompt_evo.experiments.runner import eval_on_test, save_run
    from prompt_evo.logging.operator_logger import format_stats_table
    cfg, task, worker, opt, _ = _setup(args)
    out = _out_dir(cfg, args)
    from prompt_evo.logging.evolution_logger import EvolutionLogger
    res = PromptEvo(cfg, task, worker, opt, EvolutionLogger(out / "evolution.jsonl")).run()
    test_acc, _ = eval_on_test(res.best.text, task, worker, cfg)
    save_run(res, out, cfg, task, test_acc)
    print(f"seed dev accuracy : {res.seed_individual.accuracy:.3f}\nbest dev accuracy : {res.best.accuracy:.3f}"
          f"\ntest accuracy     : {test_acc}\nworker/optimizer calls: {res.budget['worker_calls']}/{res.budget['optimizer_calls']}")
    print("\nBEST PROMPT:\n" + res.best.text + "\n\nOPERATOR ANALYSIS:\n" + format_stats_table(res.operator_stats["all_evaluated"]))
    print(f"\nresults written to {out}")
    return 0


def cmd_evaluate(args) -> int:
    from prompt_evo.core.budget import BudgetManager
    from prompt_evo.evaluation.evaluator import Evaluator
    from prompt_evo.evaluation.metrics import METRICS
    from prompt_evo.llm.factory import build_worker
    from prompt_evo.schemas.dataset import load_dataset
    cfg = load_config(args.config)
    ev = Evaluator(build_worker(cfg.worker), METRICS[args.task_type or cfg.task.type], BudgetManager(None))
    r = ev.evaluate(Path(args.prompt).read_text().strip(), load_dataset(args.dataset))
    print(json.dumps({"accuracy": r.accuracy, "correct": r.n_correct, "incorrect": r.n_incorrect, "worker_calls": r.worker_calls,
                      "latency_s": round(r.latency, 3), "cost": r.estimated_cost}, indent=2))
    return 0


def cmd_baseline(args) -> int:
    from prompt_evo.experiments.runner import run_method
    cfg, task, worker, opt, _ = _setup(args)
    mr, rr = run_method(args.method, cfg, task, worker, opt, args.seed or cfg.optimizer.seed,
                        run_dir=_out_dir(cfg, args, f"baseline_{args.method}") if args.method == "promptevo" else None)
    print(mr.model_dump_json(indent=2, exclude={"test_correctness", "extra"}))
    return 0


def cmd_experiment(args) -> int:
    from prompt_evo.experiments import ablation, robustness, runner
    cfg, task, worker, opt, base = _setup(args)
    out, kind = _out_dir(cfg, args), args.kind
    if kind in ("A", "all"):
        print(runner.format_summary(runner.run_experiment_a(cfg, task, worker, opt, out / "A")))
    if kind in ("B1", "all"):
        for r in ablation.run_b1(cfg, task, worker, opt, out / "B1"):
            print(f"B1 {r['operator']:<22} gain {r['gain_mean']:+.3f}")
    if kind in ("B2", "all"):
        for r in ablation.run_b2(cfg, task, worker, opt, out / "B2"):
            print(f"B2 without {r['removed_operator']:<22} contribution {r['operator_contribution']:+.3f}")
    if kind in ("C", "all"):
        print(json.dumps(robustness.run_c(cfg, opt, out / "C", base)["best_operator_varies_by"]))
    if kind == "transfer" or (kind == "all" and cfg.experiment.transfer_worker):
        print(json.dumps(robustness.run_transfer(cfg, opt, out / "transfer", base)["rows"], indent=1))
    print(f"results written to {out}")
    return 0


def cmd_analyze(args) -> int:
    from prompt_evo.logging.operator_logger import compute_operator_stats, format_stats_table, load_operator_log
    root = Path(args.results)
    logs = sorted(root.rglob("operator_log.jsonl"))
    if not logs:
        print(f"no operator_log.jsonl under {root}", file=sys.stderr)
        return 1
    entries = [e for p in logs for e in load_operator_log(p)]
    stats = compute_operator_stats(entries, only_full=args.full_only)
    print(f"{len(logs)} run(s), {len(entries)} children\n" + format_stats_table(stats))
    (root / "analysis_operator_stats.json").write_text(json.dumps(stats, indent=2))
    if args.plots:
        from prompt_evo.viz import make_run_plots
        for p in logs:
            if (p.parent / "summary.json").exists() and (p.parent / "evolution.jsonl").exists():
                make_run_plots(p.parent)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="prompt_evo", description="PromptEvo: black-box evolutionary prompt optimizer (frozen worker LLM).")
    ap.add_argument("-v", "--verbose", action="store_true", help="log evolution events")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("optimize", help="run PromptEvo and write results"); p.add_argument("--config", required=True); p.add_argument("--output")
    p.set_defaults(fn=cmd_optimize)
    p = sub.add_parser("evaluate", help="evaluate a prompt file on a dataset with the configured worker")
    p.add_argument("--prompt", required=True); p.add_argument("--dataset", required=True)
    p.add_argument("--config", default="configs/example.yaml"); p.add_argument("--task-type", choices=["classification", "math", "qa"])
    p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("baseline", help="run one method: seed|manual|genetic|promptevo")
    p.add_argument("--method", required=True, choices=["seed", "manual", "genetic", "promptevo"]); p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int); p.add_argument("--output"); p.set_defaults(fn=cmd_baseline)
    p = sub.add_parser("experiment", help="run experiments A, B1, B2, C, transfer or all")
    p.add_argument("--config", required=True); p.add_argument("--kind", default="A", choices=["A", "B1", "B2", "C", "transfer", "all"])
    p.add_argument("--output"); p.set_defaults(fn=cmd_experiment)
    p = sub.add_parser("analyze", help="aggregate operator statistics from saved results")
    p.add_argument("--results", required=True); p.add_argument("--full-only", action="store_true"); p.add_argument("--plots", action="store_true")
    p.set_defaults(fn=cmd_analyze)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
