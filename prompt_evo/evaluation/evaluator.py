"""Budget-aware evaluator with caching, racing and timeouts."""
from __future__ import annotations

import time
from typing import Sequence

from prompt_evo.core.budget import BudgetExhausted, BudgetManager
from prompt_evo.evaluation.metrics import Metric, extract_answer
from prompt_evo.evaluation.racing import RacingPolicy, should_stop
from prompt_evo.llm.base import LLMError, LLMResponse, WorkerModel
from prompt_evo.schemas.dataset import Example
from prompt_evo.schemas.evaluation import EvaluationResult, ExampleResult
from prompt_evo.utils import sha1_text


class EvaluationError(RuntimeError):
    """Worker failure during evaluation. Evaluation data is never silently corrupted."""


class Evaluator:
    """Evaluates a prompt on examples with a frozen worker.

    Each fresh worker call is reserved from the global BudgetManager before it is made.
    Results are cached per (prompt, example) when ``cache`` is on (valid for temperature 0),
    so staged/promoted evaluations never pay twice for the same pair.
    """

    def __init__(self, worker: WorkerModel, metric: Metric, budget: BudgetManager, *,
                 cache: bool = True, timeout_s: float | None = None) -> None:
        self.worker, self.metric, self.budget = worker, metric, budget
        self.use_cache, self.timeout_s = cache, timeout_s
        self._cache: dict[tuple[str, str], ExampleResult] = {}

    def _score(self, ex: Example, resp: LLMResponse, latency: float) -> ExampleResult:
        out = resp.text or ""
        ok = bool(out.strip()) and self.metric(out, ex.expected_output)
        cat = None
        if not ok:
            cat = "empty_response" if not out.strip() else ("wrong_answer" if extract_answer(out)[1] else "no_answer_marker")
        return ExampleResult(example_id=ex.id, input=ex.input, expected_output=ex.expected_output,
                             model_output=out, correct=ok, category=cat, latency=latency)

    def evaluate(self, prompt_text: str, examples: Sequence[Example], *, reference: float | None = None,
                 racing: RacingPolicy | None = None) -> EvaluationResult:
        h, total, t0 = sha1_text(prompt_text), len(examples), time.time()
        results: list[ExampleResult] = []
        n_correct = calls = hits = pt = ct = 0
        stop: str | None = None
        for ex in examples:
            key = (h, ex.id)
            if self.use_cache and key in self._cache:
                r = self._cache[key]
                hits += 1
            else:
                if self.timeout_s is not None and time.time() - t0 > self.timeout_s:
                    stop = "timeout"
                    break
                try:
                    self.budget.reserve_worker()
                except BudgetExhausted:
                    stop = "budget"
                    break
                calls += 1
                t1 = time.time()
                try:
                    resp = self.worker.generate(prompt_text, ex.input)
                except LLMError as e:
                    raise EvaluationError(f"worker failed on example {ex.id}: {e}") from e
                self.budget.add_tokens("worker", resp.prompt_tokens, resp.completion_tokens)
                pt, ct = pt + resp.prompt_tokens, ct + resp.completion_tokens
                r = self._score(ex, resp, time.time() - t1)
                if self.use_cache:
                    self._cache[key] = r
            results.append(r)
            n_correct += int(r.correct)
            if racing is not None and reference is not None and should_stop(racing, len(results), n_correct, total, reference):
                stop = "racing"
                break
        n = len(results)
        return EvaluationResult(
            prompt_hash=h, accuracy=n_correct / n if n else 0.0, n_correct=n_correct, n_incorrect=n - n_correct,
            n_evaluated=n, n_requested=total, complete=stop is None, stop_reason=stop, per_example=results,
            latency=time.time() - t0, worker_calls=calls, cache_hits=hits, prompt_tokens=pt, completion_tokens=ct,
            estimated_cost=self.budget.cost_of("worker", pt, ct),
        )
