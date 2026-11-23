import json
import pytest
from prompt_evo.baselines.genetic import run_genetic
from prompt_evo.baselines.manual import run_manual
from prompt_evo.baselines.seed import run_seed
from prompt_evo.experiments import runner
from prompt_evo.experiments.significance import cohens_dz, compare_paired, paired_permutation_test
from tests.fakes import ToyOptimizer, ToyWorker


def test_significance():
    a, b = [1.0] * 20 + [0.0] * 5, [0.0] * 25
    assert paired_permutation_test(a, b, 5000) < 0.01
    assert paired_permutation_test([1, 0, 1], [1, 0, 1]) == 1.0
    assert cohens_dz([2, 3, 4], [1, 1, 1]) > 1
    assert compare_paired(a, b)["mean_diff"] == pytest.approx(0.8)


def test_baselines(cfg, task):
    s = run_seed(cfg, task, ToyWorker(), 1)
    assert s.dev_accuracy is not None and s.worker_calls == len(task.dataset)
    assert run_manual(cfg, task, ToyWorker(), 1).available is False  # not invented
    task.manual_prompt = "Classify.\nConstraints:\n- Answer with only the final answer, no extra words."
    assert run_manual(cfg, task, ToyWorker(), 1).available
    g = run_genetic(cfg, task, ToyWorker(), ToyOptimizer(), 1)
    assert g.worker_calls <= cfg.optimizer.worker_budget and g.dev_accuracy >= s.dev_accuracy


def test_experiment_a_end_to_end(cfg, task, tmp_path):
    cfg.experiment.seeds = [1, 2, 3]
    task.manual_prompt = "Classify.\nConstraints:\n- Answer with only the final answer, no extra words."
    out = runner.run_experiment_a(cfg, task, ToyWorker(), ToyOptimizer(), tmp_path)
    assert {r["method"] for r in out["rows"]} == {"seed", "manual", "genetic", "promptevo"}
    assert "promptevo_vs_seed" in out["paired_tests_over_test_examples"]
    assert (tmp_path / "summary.json").exists() and (tmp_path / "plots" / "baseline_comparison.png").exists()
    run = tmp_path / "runs" / "promptevo_seed1"
    for f in ["config.json", "evolution.jsonl", "operator_stats.json", "final_population.json", "summary.json", "per_example_results.json", "operator_log.csv"]:
        assert (run / f).exists(), f
    meta = json.loads((run / "config.json").read_text())
    assert meta["random_seed"] == 1 and meta["dataset_hash"] and "worker_model" in meta and "git_commit" in meta
    assert len(list((run / "plots").glob("*.png"))) >= 6


def test_ablation_and_robustness(cfg, task, tmp_path):
    from prompt_evo.experiments.ablation import run_b1, run_b2
    cfg.experiment.seeds = [1, 2]
    cfg.optimizer.operators = ["constraints", "reasoning"]
    b1 = run_b1(cfg, task, ToyWorker(), ToyOptimizer(), tmp_path / "b1")
    assert {r["operator"] for r in b1} == {"constraints", "reasoning"}
    b2 = run_b2(cfg, task, ToyWorker(), ToyOptimizer(), tmp_path / "b2")
    assert {r["removed_operator"] for r in b2} == {"constraints", "reasoning"} and "operator_contribution" in b2[0]


def test_test_set_not_used_for_selection(cfg, task):
    from prompt_evo.core.optimizer import PromptEvo
    task.test_dataset = []
    r = PromptEvo(cfg, task, ToyWorker(), ToyOptimizer()).run()  # runs fine with no test set at all
    assert r.best.evaluation.n_evaluated == len(task.dataset)
