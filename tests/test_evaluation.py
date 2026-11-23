import pytest
from prompt_evo.core.budget import BudgetExhausted, BudgetManager
from prompt_evo.evaluation.bootstrap import bootstrap_accuracy, paired_bootstrap_delta
from prompt_evo.evaluation.evaluator import EvaluationError, Evaluator
from prompt_evo.evaluation.metrics import METRICS, extract_answer
from prompt_evo.evaluation.racing import RacingPolicy, should_stop
from prompt_evo.evaluation.slices import make_slices, stage_sizes
from prompt_evo.llm.base import LLMError, LLMResponse, WorkerModel
from tests.fakes import ToyWorker


def test_extract_answer_variants():
    assert extract_answer("thinking\nAnswer: 42") == ("42", True)
    assert extract_answer('{"answer": "yes"}') == ("yes", True)
    assert extract_answer("just text")[1] is False


def test_metrics():
    assert METRICS["classification"]("Answer: Positive.", "positive")
    assert not METRICS["classification"]("negative", "positive")
    assert METRICS["math"]("so 1,234 total\nAnswer: $1,234", "1234") and METRICS["math"]("Answer: 3/4", "0.75")
    assert METRICS["qa"]("Answer: The city of Paris", "Paris") and not METRICS["qa"]("Answer: Lyon", "Paris")


def test_evaluator_basics_and_cache(dev):
    b = BudgetManager(1000)
    ev = Evaluator(ToyWorker(), METRICS["classification"], b)
    r = ev.evaluate("Answer: <answer> only the final answer", dev[:20])
    assert r.n_correct + r.n_incorrect == 20 and r.worker_calls == 20 and b.worker_calls == 20
    assert all(x.expected_output and x.model_output for x in r.per_example)
    r2 = ev.evaluate("Answer: <answer> only the final answer", dev[:30])  # first 20 cached
    assert r2.worker_calls == 10 and r2.cache_hits == 20 and b.worker_calls == 30


def test_budget_never_exceeded(dev):
    b = BudgetManager(25)
    r = Evaluator(ToyWorker(), METRICS["classification"], b).evaluate("p", dev[:60])
    assert not r.complete and r.stop_reason == "budget" and b.worker_calls == 25 and r.n_evaluated == 25
    with pytest.raises(BudgetExhausted):
        b.reserve_worker()


def test_budget_accounting_and_cost():
    from prompt_evo.core.budget import Pricing
    b = BudgetManager(10, 5, Pricing(1.0, 2.0), Pricing(3.0, 0.0))
    b.reserve_worker(); b.reserve_optimizer()
    b.add_tokens("worker", 1000, 1000); b.add_tokens("optimizer", 1000, 0)
    assert (b.worker_calls, b.optimizer_calls, b.total_calls, b.worker_remaining) == (1, 1, 2, 9)
    assert b.total_cost == pytest.approx(1 + 2 + 3)


def test_worker_failure_raises(dev):
    class Bad(WorkerModel):
        def generate(self, prompt, input):
            raise LLMError("boom")
    with pytest.raises(EvaluationError):
        Evaluator(Bad(), METRICS["classification"], BudgetManager(10)).evaluate("p", dev[:3])


def test_empty_response_marked_incorrect(dev):
    class Empty(WorkerModel):
        def generate(self, prompt, input):
            return LLMResponse(text="")
    r = Evaluator(Empty(), METRICS["classification"], BudgetManager(10)).evaluate("p", dev[:3])
    assert r.n_correct == 0 and r.per_example[0].category == "empty_response"


def test_racing_bound_rule():
    pol = RacingPolicy()
    assert should_stop(pol, 10, 2, 20, 0.9)        # max 12/20=0.6 < 0.9
    assert not should_stop(pol, 10, 9, 20, 0.9)    # max 19/20
    assert not should_stop(RacingPolicy(enabled=False), 10, 0, 20, 0.9)
    assert should_stop(RacingPolicy(use_bound=False, use_confidence=True, min_examples=10), 30, 3, 100, 0.8)


def test_racing_saves_calls(dev):
    b = BudgetManager(None)
    ev = Evaluator(ToyWorker(), METRICS["classification"], b)
    r = ev.evaluate("terrible prompt", dev, reference=0.999, racing=RacingPolicy())
    assert r.stop_reason == "racing" and b.worker_calls < len(dev)


def test_bootstrap():
    r = bootstrap_accuracy([1] * 70 + [0] * 30, 200, seed=1)
    assert 0.6 < r.ci_low < r.mean < r.ci_high < 0.8 and r.std > 0
    assert bootstrap_accuracy([1] * 70 + [0] * 30, 200, seed=1) == r
    d = paired_bootstrap_delta([1] * 80 + [0] * 20, [1] * 50 + [0] * 50)
    assert d["mean"] == pytest.approx(0.3) and d["p_positive"] > 0.99


def test_slices_nested_and_reproducible(dev):
    sizes = stage_sizes(20, 50, len(dev))
    s1, s2 = make_slices(dev, sizes, 7), make_slices(dev, sizes, 7)
    assert sizes == [20, 50] and [[e.id for e in s] for s in s1] == [[e.id for e in s] for s in s2]
    assert {e.id for e in s1[0]} < {e.id for e in s1[1]} < {e.id for e in s1[2]} and len(s1[2]) == len(dev)
    assert stage_sizes(20, 50, 30) == [20]
