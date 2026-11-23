"""PromptEvo main evolutionary loop (steady-state, budget-aware)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from prompt_evo.config import Config
from prompt_evo.core.budget import BudgetExhausted, BudgetManager, Pricing
from prompt_evo.core.individual import Individual
from prompt_evo.core.operator_selector import FixedSelector, OperatorSelector, ThompsonSelector
from prompt_evo.core.population import Population
from prompt_evo.core.selection import Selection
from prompt_evo.evaluation.bootstrap import bootstrap_accuracy
from prompt_evo.evaluation.evaluator import EvaluationError, Evaluator
from prompt_evo.evaluation.metrics import METRICS
from prompt_evo.evaluation.racing import RacingPolicy
from prompt_evo.evaluation.slices import make_slices, stage_sizes
from prompt_evo.llm.base import OptimizerModel, WorkerModel
from prompt_evo.logging.evolution_logger import EvolutionLogger
from prompt_evo.logging.operator_logger import OperatorLogger
from prompt_evo.operators import CrossoverOperator, FeedbackOperator, InvalidMutation, Operator, OperatorContext, build_operators
from prompt_evo.prompts.parser import parse_prompt
from prompt_evo.prompts.serializer import prompt_hash, serialize_prompt
from prompt_evo.schemas.dataset import TaskConfig
from prompt_evo.schemas.evaluation import EvaluationResult
from prompt_evo.schemas.evolution import OperatorLogEntry
from prompt_evo.schemas.prompt import StructuredPrompt

MAX_CONSECUTIVE_WORKER_FAILURES = 5


@dataclass
class RunResult:
    best: Individual
    seed_individual: Individual
    population: list[Individual]
    history: list[dict[str, Any]]
    operator_log: OperatorLogger
    operator_stats: dict[str, list[dict]]
    budget: dict[str, Any]
    selector_state: dict[str, Any] = field(default_factory=dict)
    evolution: EvolutionLogger | None = None


def pricing_of(section) -> Pricing:
    return Pricing(section.price_per_1k_prompt, section.price_per_1k_completion)


class PromptEvo:
    """Evolves a StructuredPrompt against a frozen worker. Only prompts change."""

    def __init__(self, cfg: Config, task: TaskConfig, worker: WorkerModel, optimizer_llm: OptimizerModel,
                 evolution_logger: EvolutionLogger | None = None) -> None:
        o, e = cfg.optimizer, cfg.evaluation
        self.cfg, self.task, self.o, self.e = cfg, task, o, e
        self.rng = random.Random(o.seed)
        self.budget = BudgetManager(o.worker_budget, o.max_optimizer_calls, pricing_of(cfg.worker), pricing_of(cfg.optimizer_llm))
        self.evaluator = Evaluator(worker, METRICS[task.evaluator], self.budget, cache=e.cache, timeout_s=e.eval_timeout_s)
        self.racing = RacingPolicy(**e.racing.model_dump(exclude={"reference"}))
        self.stages = make_slices(task.dataset, stage_sizes(e.initial_slice_size, e.promotion_slice_size, len(task.dataset)), o.seed)
        self.operators: list[Operator] = build_operators(o.operators)
        if not self.operators and o.crossover_probability <= 0 and o.feedback_probability <= 0:
            raise ValueError("no operators enabled")
        self.crossover, self.feedback = CrossoverOperator(), FeedbackOperator()
        names = [op.name for op in self.operators]
        self.selector: OperatorSelector = ThompsonSelector(names) if o.adaptive else FixedSelector(names, o.operator_weights)
        self.selection = Selection(o.selection, o.tournament_size, o.robust_penalty)
        self.pop = Population()
        self.oplog = OperatorLogger()
        self.evo = evolution_logger or EvolutionLogger()
        self.ctx = OperatorContext(task.description, optimizer_llm, self.budget, self.rng,
                                   pool=task.fewshot_pool or task.dataset, max_retries=o.max_retries, max_fewshot=o.max_fewshot)
        self.history: list[dict[str, Any]] = []
        self._worker_failures = 0
        self.seed_ind: Individual | None = None

    # ---------- helpers ----------
    def _fitness(self, acc: float, length: int) -> float:
        return acc - self.e.length_penalty * length / self.e.max_prompt_chars

    def _individual(self, prompt: StructuredPrompt, text: str, res: EvaluationResult) -> Individual:
        h = prompt_hash(prompt)
        boot = bootstrap_accuracy([r.correct for r in res.per_example], self.e.bootstrap_samples, self.o.seed + int(h[:6], 16) % 10_000)
        return Individual(id=prompt.version_id, prompt=prompt, text=text, hash=h, accuracy=res.accuracy,
                          fitness=self._fitness(res.accuracy, len(text)), length=len(text), generation=prompt.generation,
                          parent_ids=prompt.parent_ids, operator=prompt.operator, evaluation=res, bootstrap=boot)

    def _reference(self, parents: list[Individual], subset) -> float:
        ids = [x.id for x in subset]
        pool = [self.selection.best(self.pop.individuals)] if self.e.racing.reference == "best" and self.pop.individuals else parents
        return max(sum(p.evaluation.correctness()[i] for i in ids) / len(ids) for p in pool)

    def _record_point(self) -> None:
        best = self.selection.best(self.pop.individuals)
        self.history.append({"worker_calls": self.budget.worker_calls, "optimizer_calls": self.budget.optimizer_calls,
                             "best_accuracy": best.accuracy, "best_fitness": best.fitness, "best_id": best.id,
                             "population_mean_accuracy": self.pop.mean_accuracy(), "population_size": len(self.pop)})

    # ---------- evaluation ----------
    def _evaluate_child(self, text: str, parents: list[Individual]) -> tuple[EvaluationResult, str]:
        tol = self.e.promote_tolerance
        res: EvaluationResult | None = None
        for si, subset in enumerate(self.stages):
            ref = self._reference(parents, subset)
            res = self.evaluator.evaluate(text, subset, reference=ref, racing=self.racing)
            if not res.complete:
                return res, ("budget_exhausted" if res.stop_reason in ("budget", "timeout") else "racing_stopped")
            if si < len(self.stages) - 1 and res.accuracy < ref - tol:
                return res, "not_promoted"
        assert res is not None
        return res, "evaluated"

    # ---------- logging ----------
    def _log_entry(self, child_id: str, parents: list[Individual], op_name: str, gen: int, status: str, snap: dict,
                   *, text: str = "", res: EvaluationResult | None = None, accepted: bool = False,
                   rec=None, reason: str = "") -> OperatorLogEntry:
        ref = max(parents, key=lambda p: p.fitness)
        e = OperatorLogEntry(child_id=child_id, parent_ids=[p.id for p in parents], generation=gen, operator=op_name,
                             status=status, before_accuracy=ref.accuracy, before_score=ref.fitness,
                             prompt_length_before=ref.length, prompt_length_after=len(text), accepted=accepted,
                             worker_calls=self.budget.worker_calls - snap["worker_calls"],
                             optimizer_calls=self.budget.optimizer_calls - snap["optimizer_calls"],
                             change_description=(rec.reason if rec else reason),
                             component_before=rec.original_component if rec else None,
                             component_after=rec.new_component if rec else None,
                             llm_call_ids=rec.llm_call_ids if rec else [])
        if res is not None and res.per_example:
            pc, cc = ref.evaluation.correctness(), res.correctness()
            ids = list(cc)
            e.examples_fixed = [i for i in ids if cc[i] and not pc[i]]
            e.examples_broken = [i for i in ids if not cc[i] and pc[i]]
            e.examples_unchanged = len(ids) - len(e.examples_fixed) - len(e.examples_broken)
            e.n_evaluated, e.full_eval = len(ids), res.complete and len(ids) == len(self.task.dataset)
            e.before_accuracy = sum(pc[i] for i in ids) / len(ids)
            e.after_accuracy = res.accuracy
            e.accuracy_delta = e.after_accuracy - e.before_accuracy
            e.before_score = self._fitness(e.before_accuracy, ref.length)
            e.after_score = self._fitness(res.accuracy, len(text))
            e.score_delta = e.after_score - e.before_score
        self.oplog.add(e)
        self.evo.log("child", child_id=child_id, operator=op_name, status=status, generation=gen,
                     score_delta=e.score_delta, accuracy_delta=e.accuracy_delta, accepted=accepted,
                     worker_calls_used=self.budget.worker_calls, optimizer_calls_used=self.budget.optimizer_calls)
        return e

    # ---------- child production ----------
    def _produce(self, op: Operator, parents: list[Individual], gen: int) -> tuple[Individual | None, OperatorLogEntry]:
        snap = self.budget.snapshot()
        cid = self.pop.next_id()
        try:
            out = op.apply([p.prompt for p in parents], self.ctx)
        except InvalidMutation as ex:
            return None, self._log_entry(cid, parents, op.name, gen, "invalid_mutation", snap, reason=str(ex)[:300])
        child = out.child.model_copy(update={"version_id": cid, "parent_ids": [p.id for p in parents], "operator": op.name,
                                              "generation": gen, "metadata": {"reason": out.record.reason}})
        text, h = serialize_prompt(child), prompt_hash(child)
        if self.pop.is_duplicate(h):
            return None, self._log_entry(cid, parents, op.name, gen, "duplicate", snap, text=text, rec=out.record)
        self.pop.register(h)
        if len(text) > self.e.hard_max_prompt_chars:
            return None, self._log_entry(cid, parents, op.name, gen, "too_long", snap, text=text, rec=out.record)
        try:
            res, status = self._evaluate_child(text, parents)
        except EvaluationError as ex:
            self._worker_failures += 1
            if self._worker_failures >= MAX_CONSECUTIVE_WORKER_FAILURES:
                raise RuntimeError(f"aborting: {self._worker_failures} consecutive worker failures ({ex})") from ex
            return None, self._log_entry(cid, parents, op.name, gen, "worker_failure", snap, text=text, rec=out.record, reason=str(ex)[:300])
        self._worker_failures = 0
        if status != "evaluated":
            return None, self._log_entry(cid, parents, op.name, gen, status, snap, text=text, res=res, rec=out.record)
        ind = self._individual(child, text, res)
        self.pop.add(ind)
        survivors = self.selection.survivors(self.pop.individuals, self.o.population_size)
        accepted = any(s.id == ind.id for s in survivors)
        self.pop.individuals = survivors
        return ind, self._log_entry(cid, parents, op.name, gen, "evaluated", snap, text=text, res=res, accepted=accepted, rec=out.record)

    # ---------- stages of the run ----------
    def _init_seed(self) -> None:
        p = parse_prompt(self.task.seed_prompt, version_id=self.pop.next_id())
        text = serialize_prompt(p)
        self.pop.register(prompt_hash(p))
        res = self.evaluator.evaluate(text, self.stages[-1])
        if not res.complete:
            raise BudgetExhausted("worker budget too small to evaluate the seed prompt on the dev set")
        self.seed_ind = self._individual(p, text, res)
        self.pop.add(self.seed_ind)
        self.evo.log("init_seed", id=p.version_id, accuracy=res.accuracy, fitness=self.seed_ind.fitness, text=text)
        self._record_point()

    def _parents_for(self, kind: str) -> list[Individual]:
        inds = self.pop.individuals
        if kind == "crossover":
            a = self.selection.pick_parent(inds, self.rng)
            others = [i for i in inds if i.id != a.id]
            b = self.selection.pick_parent(others, self.rng)
            return [a, b]
        return [self.selection.pick_parent(inds, self.rng)]

    def _choose_kind(self, init: bool = False) -> str:
        o = self.o
        w = {"mutation": o.mutation_probability if self.operators else 0.0,
             "crossover": o.crossover_probability if len(self.pop) >= 2 else 0.0,
             "feedback": o.feedback_probability if (not init and any(not r.correct for i in self.pop.individuals for r in i.evaluation.per_example)) else 0.0}
        if sum(w.values()) <= 0:
            w = {"mutation": 1.0, "crossover": 0.0, "feedback": 0.0} if self.operators else {"crossover": 1.0}
        return self.rng.choices(list(w), weights=list(w.values()), k=1)[0]

    def _step(self, gen: int, init: bool = False) -> None:
        kind = "mutation" if (init and len(self.pop) < 2) else self._choose_kind(init)
        if kind == "mutation":
            chosen = self.selector.choose(self.rng)
            op = next(x for x in self.operators if x.name == chosen)
            parents = [self.seed_ind] if init else self._parents_for("mutation")
        elif kind == "crossover":
            op, parents = self.crossover, self._parents_for("crossover")
        else:
            op, parents = self.feedback, self._parents_for("mutation")
            fails = [r for r in parents[0].evaluation.per_example if not r.correct]
            fails = self.rng.sample(fails, min(5, len(fails)))
            self.ctx.failures = [{"input": r.input, "expected": r.expected_output, "model_output": r.model_output[:300]} for r in fails]
        ind, entry = self._produce(op, parents, gen)
        if kind == "mutation":
            self.selector.update(op.name, 1.0 if (entry.score_delta or 0) > 1e-9 else 0.0)
        self._record_point()

    def run(self) -> RunResult:
        o = self.o
        self._init_seed()
        tries, max_tries = 0, o.population_size * o.init_attempt_factor
        while len(self.pop) < o.population_size and tries < max_tries and self.budget.worker_remaining > 0 and self.budget.optimizer_remaining > 0:
            self._step(0, init=True)
            tries += 1
        self.evo.log("init_done", population_size=len(self.pop), attempts=tries, worker_calls_used=self.budget.worker_calls)
        gen, made, per_gen = 1, 0, o.children_per_generation or o.population_size
        while gen <= o.max_generations and self.budget.worker_remaining > 0 and self.budget.optimizer_remaining > 0:
            self._step(gen)
            made += 1
            if made % per_gen == 0:
                self.evo.log("generation", generation=gen, accuracies=[i.accuracy for i in self.pop.individuals],
                             best_accuracy=self.selection.best(self.pop.individuals).accuracy,
                             worker_calls_used=self.budget.worker_calls)
                gen += 1
        best = self.selection.best(self.pop.individuals)
        self.evo.log("done", best_id=best.id, best_accuracy=best.accuracy, budget=self.budget.snapshot())
        return RunResult(best=best, seed_individual=self.seed_ind, population=list(self.pop.individuals), history=self.history,
                         operator_log=self.oplog, operator_stats={"all_evaluated": self.oplog.stats(False), "full_eval_only": self.oplog.stats(True)},
                         budget=self.budget.snapshot(), selector_state=self.selector.state(), evolution=self.evo)
