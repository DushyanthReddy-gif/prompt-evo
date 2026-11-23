import json
import random

import pytest
from prompt_evo.core.individual import Individual
from prompt_evo.core.operator_selector import FixedSelector, ThompsonSelector
from prompt_evo.core.optimizer import PromptEvo
from prompt_evo.core.selection import Selection, dominates, pareto_ranks
from prompt_evo.evaluation.bootstrap import BootstrapResult
from prompt_evo.logging.operator_logger import compute_operator_stats
from prompt_evo.prompts.parser import parse_prompt
from prompt_evo.schemas.dataset import DatasetError, load_dataset
from prompt_evo.schemas.evaluation import EvaluationResult
from prompt_evo.schemas.evolution import OperatorLogEntry
from tests.fakes import ToyOptimizer, ToyWorker


def ind(i, acc, length=100, std=0.01):
    ev = EvaluationResult(prompt_hash="h", accuracy=acc, n_correct=0, n_incorrect=0, n_evaluated=0, n_requested=0, complete=True)
    return Individual(id=i, prompt=parse_prompt("x"), text="x", hash=i, accuracy=acc, fitness=acc, length=length, generation=0,
                      evaluation=ev, bootstrap=BootstrapResult(mean=acc, std=std, ci_low=acc, ci_high=acc, n_samples=10))


def test_tournament_prefers_fitter_and_is_seeded():
    pop = [ind("a", 0.5), ind("b", 0.9), ind("c", 0.6)]
    sel = Selection("tournament", 3)
    assert sel.pick_parent(pop, random.Random(0)).id == "b"
    assert sel.best(pop).id == "b" and [i.id for i in sel.survivors(pop, 2)] == ["b", "c"]


def test_robust_score_penalizes_noise():
    sel = Selection("tournament", 2, robust_penalty=1.0)
    assert sel.best([ind("noisy", 0.80, std=0.2), ind("stable", 0.75, std=0.01)]).id == "stable"


def test_pareto():
    a, b, c = ind("a", 0.9, 200), ind("b", 0.8, 100), ind("c", 0.7, 300)
    assert dominates(a, c) and not dominates(a, b) and pareto_ranks([a, b, c])["c"] >= 1
    assert {i.id for i in Selection("pareto").survivors([a, b, c], 2)} == {"a", "b"}


def test_operator_selectors():
    names = ["x", "y"]
    assert FixedSelector(names, {"x": 0.0, "y": 1.0}).choose(random.Random(1)) == "y"
    ts, rng = ThompsonSelector(names), random.Random(0)
    for _ in range(30):
        ts.update("x", 1.0); ts.update("y", 0.0)
    assert sum(ts.choose(rng) == "x" for _ in range(50)) > 40


def test_dataset_loading_and_validation(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"id":"e1","input":"i","expected_output":"o"}\n')
    (tmp_path / "a.csv").write_text("id,input,expected_output\ne1,i,o\n")
    (tmp_path / "a.json").write_text('[{"input":"i","expected_output":3}]')
    assert load_dataset(tmp_path / "a.jsonl")[0].id == "e1" and load_dataset(tmp_path / "a.csv")[0].input == "i"
    assert load_dataset(tmp_path / "a.json")[0].expected_output == "3"
    (tmp_path / "dup.jsonl").write_text('{"id":"e","input":"i","expected_output":"o"}\n' * 2)
    (tmp_path / "bad.jsonl").write_text('{"input":"i"}\n')
    for f in ("dup.jsonl", "bad.jsonl", "missing.jsonl"):
        with pytest.raises(DatasetError):
            load_dataset(tmp_path / f)


def _entry(op, delta, fixed=(), broken=(), full=True, status="evaluated"):
    return OperatorLogEntry(child_id="c", parent_ids=["p0"], generation=1, operator=op, status=status, score_delta=delta,
                            accuracy_delta=delta, examples_fixed=list(fixed), examples_broken=list(broken), full_eval=full)


def test_operator_stats_math():
    es = [_entry("a", 0.1, ["e1", "e2"], ["e3"]), _entry("a", -0.05), _entry("a", None, status="invalid_mutation"), _entry("b", 0.0)]
    es[2].score_delta = None
    st = {r["operator"]: r for r in compute_operator_stats(es)}
    a = st["a"]
    assert a["attempts"] == 3 and a["evaluated"] == 2 and a["n_wins"] == 1 and a["win_rate"] == 0.5
    assert a["mean_score_delta"] == pytest.approx(0.025) and a["success_rate"] == pytest.approx(2 / 3)
    assert a["avg_examples_fixed"] == 1.0 and a["avg_examples_broken"] == 0.5
    assert compute_operator_stats(es)[0]["operator"] == "a"  # ranked by mean delta


def run_once(cfg, task, seed=42, **ov):
    c = cfg.model_copy(deep=True)
    c.optimizer.seed = seed
    for k, v in ov.items():
        setattr(c.optimizer, k, v)
    return PromptEvo(c, task, ToyWorker(), ToyOptimizer()).run()


def test_end_to_end_budget_and_logging(cfg, task):
    r = run_once(cfg, task)
    assert r.budget["worker_calls"] <= cfg.optimizer.worker_budget
    assert r.best.accuracy >= r.seed_individual.accuracy - 0.1 and len(r.population) <= cfg.optimizer.population_size
    assert len({i.hash for i in r.population}) == len(r.population)  # no duplicate prompts
    log = r.operator_log.entries
    assert log and {"child_id", "parent_ids", "generation", "operator", "before_accuracy", "after_accuracy", "score_delta",
                    "examples_fixed", "examples_broken", "examples_unchanged", "prompt_length_before", "prompt_length_after",
                    "worker_calls", "optimizer_calls", "timestamp"} <= set(log[0].model_dump())
    for e in log:
        if e.status == "evaluated" and e.full_eval:
            assert e.examples_unchanged + len(e.examples_fixed) + len(e.examples_broken) == len(task.dataset)
            assert e.accuracy_delta == pytest.approx(e.after_accuracy - e.before_accuracy)
    assert sum(e.worker_calls for e in log) + len(task.dataset) == r.budget["worker_calls"]  # seed eval + children
    assert {"all_evaluated", "full_eval_only"} == set(r.operator_stats)
    assert r.best.bootstrap is not None and r.best.accuracy == r.best.evaluation.accuracy  # raw accuracy preserved


def test_reproducible_with_same_seed(cfg, task):
    a, b = run_once(cfg, task, 7), run_once(cfg, task, 7)
    assert a.best.hash == b.best.hash and a.best.accuracy == b.best.accuracy
    assert [e.model_dump(exclude={"timestamp"}) for e in a.operator_log.entries] == [e.model_dump(exclude={"timestamp"}) for e in b.operator_log.entries]


def test_tiny_budget_raises_cleanly(cfg, task):
    from prompt_evo.core.budget import BudgetExhausted
    with pytest.raises(BudgetExhausted):
        run_once(cfg, task, worker_budget=10)


def test_stops_cleanly_when_budget_runs_out_midway(cfg, task):
    r = run_once(cfg, task, worker_budget=len(task.dataset) + 45, max_generations=50)
    assert r.budget["worker_calls"] <= len(task.dataset) + 45


def test_single_operator_isolation_only_uses_that_operator(cfg, task):
    r = run_once(cfg, task, operators=["constraints"], crossover_probability=0.0)
    assert {e.operator for e in r.operator_log.entries} == {"constraints"}


def test_optional_feedback_and_adaptive(cfg, task):
    r = run_once(cfg, task, feedback_probability=0.5, adaptive=True)
    assert r.selector_state["type"] == "thompson"
    assert "feedback" in {e.operator for e in r.operator_log.entries}


def test_length_penalty_keeps_raw_accuracy(cfg, task):
    c = cfg.model_copy(deep=True); c.evaluation.length_penalty = 0.5
    r = PromptEvo(c, task, ToyWorker(), ToyOptimizer()).run()
    s = r.seed_individual
    assert s.fitness == pytest.approx(s.accuracy - 0.5 * s.length / c.evaluation.max_prompt_chars) and s.fitness < s.accuracy
