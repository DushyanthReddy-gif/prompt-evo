"""Matplotlib plots. All data comes from real run artifacts; nothing is hard-coded."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)


def plot_best_vs_calls(histories: dict[str, list[dict]], path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, h in histories.items():
        ax.step([x["worker_calls"] for x in h], [x["best_accuracy"] for x in h], where="post", label=label)
    ax.set(xlabel="worker calls used", ylabel="best dev accuracy", title="Best accuracy vs worker calls")
    ax.legend()
    _save(fig, path)


def plot_population_vs_generation(events: list[dict], path) -> None:
    gens = [e for e in events if e["event"] == "generation"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for e in gens:
        ax.scatter([e["generation"]] * len(e["accuracies"]), e["accuracies"], s=12, alpha=0.5, color="tab:blue")
    if gens:
        ax.plot([e["generation"] for e in gens], [sum(e["accuracies"]) / len(e["accuracies"]) for e in gens], color="tab:red", label="mean")
        ax.legend()
    ax.set(xlabel="generation", ylabel="dev accuracy", title="Population accuracy vs generation")
    _save(fig, path)


def plot_operator_metric(stats: list[dict], key: str, title: str, path, ci: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["operator"] for r in stats]
    vals = [r[key] for r in stats]
    err = None
    if ci:
        err = [[r[key] - r["ci_low"] for r in stats], [r["ci_high"] - r[key] for r in stats]]
    ax.bar(names, vals, yerr=err, capsize=3)
    ax.set(ylabel=key, title=title)
    ax.axhline(0, color="black", lw=0.5)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, path)


def plot_length_vs_accuracy(pop: list[dict], path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([p["length"] for p in pop], [p["accuracy"] for p in pop])
    for p in pop:
        ax.annotate(p["id"], (p["length"], p["accuracy"]), fontsize=7)
    ax.set(xlabel="prompt length (chars)", ylabel="dev accuracy", title="Prompt length vs accuracy")
    _save(fig, path)


def plot_budget_performance(rows: list[dict], path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(f"{r['task']}/{r['worker']}", []).append(r)
    for label, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["budget"])
        ax.plot([r["budget"] for r in rs], [r["mean_final_accuracy"] for r in rs], marker="o", label=label)
    ax.set(xlabel="worker call budget", ylabel="mean final accuracy", title="Performance across budgets")
    ax.legend(fontsize=7)
    _save(fig, path)


def plot_baseline_comparison(rows: list[dict], path) -> None:
    rows = [r for r in rows if r.get("available")]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([r["method"] for r in rows], [r["mean"] for r in rows], yerr=[r["std"] for r in rows], capsize=4)
    ax.set(ylabel="mean accuracy (+/- std over seeds)", title="Baseline comparison")
    _save(fig, path)


def make_run_plots(run_dir: str | Path) -> None:
    d = Path(run_dir)
    plots = d / "plots"
    summary = json.loads((d / "summary.json").read_text())
    plot_best_vs_calls({"PromptEvo": summary["history"]}, plots / "best_accuracy_vs_calls.png")
    events = [json.loads(line) for line in (d / "evolution.jsonl").read_text().splitlines() if line.strip()]
    plot_population_vs_generation(events, plots / "population_accuracy_vs_generation.png")
    stats = json.loads((d / "operator_stats.json").read_text())["all_evaluated"]
    stats = [s for s in stats if s["evaluated"] > 0]
    if stats:
        plot_operator_metric(stats, "mean_score_delta", "Operator mean score delta (95% CI)", plots / "operator_mean_delta.png", ci=True)
        plot_operator_metric(stats, "win_rate", "Operator win rate", plots / "operator_win_rate.png")
        plot_operator_metric(stats, "success_rate", "Operator success rate (valid evaluated children / attempts)", plots / "operator_success_rate.png")
    plot_length_vs_accuracy(json.loads((d / "final_population.json").read_text()), plots / "length_vs_accuracy.png")


def make_experiment_plots(out: Path, rows: list[dict], results: dict[str, list[Any]]) -> None:
    plot_baseline_comparison(rows, out / "plots" / "baseline_comparison.png")
    hist = {}
    for m, rs in results.items():
        for r in rs:
            h = r.extra.get("history")
            if h:
                hist[f"{m}_seed{r.seed}"] = h
    if hist:
        plot_best_vs_calls(hist, out / "plots" / "best_accuracy_vs_calls.png")
