"""Baseline 3: basic EvoPrompt-style genetic prompt optimizer.

Whole-prompt LLM mutation/crossover, full-dev-set evaluation, truncation survivor selection by raw
accuracy. No structured slots, no operator analysis, no adaptive selection, no racing/slices/bootstrap.
"""
from __future__ import annotations

import random

from prompt_evo.config import Config
from prompt_evo.core.budget import BudgetExhausted, BudgetManager
from prompt_evo.core.optimizer import pricing_of
from prompt_evo.evaluation.evaluator import EvaluationError, Evaluator
from prompt_evo.evaluation.metrics import METRICS
from prompt_evo.llm.base import OptimizerModel, WorkerModel
from prompt_evo.operators.base import InvalidMutation, Operator, OperatorContext
from prompt_evo.prompts.crossover_templates import GENERIC_CROSSOVER, GENERIC_MUTATION
from prompt_evo.prompts.parser import parse_prompt
from prompt_evo.prompts.serializer import prompt_hash, serialize_prompt
from prompt_evo.schemas.dataset import TaskConfig
from prompt_evo.schemas.evolution import MethodResult
from prompt_evo.schemas.prompt import StructuredPrompt


class _GenericOp(Operator):
    slot = "all"

    def parse_response(self, data, parents, ctx):
        text = data["prompt"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty prompt")
        child = parse_prompt(text)
        return child, serialize_prompt(parents[0]), serialize_prompt(child), str(data.get("reason", ""))[:300], None


class GenericMutation(_GenericOp):
    name = "generic_mutation"

    def build_user_prompt(self, parents, ctx):
        return GENERIC_MUTATION.format(task=ctx.task_description, prompt=serialize_prompt(parents[0]))


class GenericCrossover(_GenericOp):
    name, n_parents = "generic_crossover", 2

    def build_user_prompt(self, parents, ctx):
        return GENERIC_CROSSOVER.format(task=ctx.task_description, a=serialize_prompt(parents[0]), b=serialize_prompt(parents[1]))


def run_genetic(cfg: Config, task: TaskConfig, worker: WorkerModel, optimizer_llm: OptimizerModel, seed: int) -> MethodResult:
    o = cfg.optimizer
    rng = random.Random(seed)
    budget = BudgetManager(o.worker_budget, o.max_optimizer_calls, pricing_of(cfg.worker), pricing_of(cfg.optimizer_llm))
    ev = Evaluator(worker, METRICS[task.evaluator], budget, cache=cfg.evaluation.cache)
    ctx = OperatorContext(task.description, optimizer_llm, budget, rng, max_retries=o.max_retries)
    mut, cross = GenericMutation(), GenericCrossover()
    seed_p = parse_prompt(task.seed_prompt)
    seen = {prompt_hash(seed_p)}
    res = ev.evaluate(serialize_prompt(seed_p), task.dataset)
    if not res.complete:
        raise BudgetExhausted("worker budget too small to evaluate the seed prompt")
    pop: list[tuple[StructuredPrompt, float, int]] = [(seed_p, res.accuracy, len(serialize_prompt(seed_p)))]
    history = [{"worker_calls": budget.worker_calls, "best_accuracy": res.accuracy}]

    def attempt(op: Operator, parents: list[StructuredPrompt]) -> None:
        try:
            out = op.apply(parents, ctx)
        except InvalidMutation:
            return
        h = prompt_hash(out.child)
        if h in seen:
            return
        seen.add(h)
        text = serialize_prompt(out.child)
        try:
            r = ev.evaluate(text, task.dataset)
        except EvaluationError:
            return
        if not r.complete:
            return
        pop.append((out.child, r.accuracy, len(text)))
        pop.sort(key=lambda t: (-t[1], t[2]))
        del pop[o.population_size:]
        history.append({"worker_calls": budget.worker_calls, "best_accuracy": pop[0][1]})

    tries = 0
    while len(pop) < o.population_size and tries < o.population_size * o.init_attempt_factor and budget.worker_remaining > 0:
        attempt(mut, [seed_p])
        tries += 1
    steps, max_steps = 0, o.max_generations * o.population_size
    while steps < max_steps and budget.worker_remaining > 0 and budget.optimizer_remaining > 0:
        steps += 1
        if len(pop) >= 2 and rng.random() < o.crossover_probability:
            a, b = rng.sample(pop, 2)
            attempt(cross, [a[0], b[0]])
        else:
            attempt(mut, [rng.choice(pop)[0]])
    best_p, best_acc, _ = pop[0]
    text = serialize_prompt(best_p)
    return MethodResult(method="genetic", seed=seed, prompt_text=text, dev_accuracy=best_acc, worker_calls=budget.worker_calls,
                        optimizer_calls=budget.optimizer_calls, prompt_length=len(text), extra={"history": history})
