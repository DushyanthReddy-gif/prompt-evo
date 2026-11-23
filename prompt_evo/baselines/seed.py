"""Baseline 1: the unmodified seed prompt (evaluated in the same normalized form all methods use)."""
from __future__ import annotations

from prompt_evo.config import Config
from prompt_evo.core.budget import BudgetManager
from prompt_evo.evaluation.evaluator import Evaluator
from prompt_evo.evaluation.metrics import METRICS
from prompt_evo.llm.base import WorkerModel
from prompt_evo.prompts.parser import parse_prompt
from prompt_evo.prompts.serializer import serialize_prompt
from prompt_evo.schemas.dataset import TaskConfig
from prompt_evo.schemas.evolution import MethodResult


def evaluate_fixed_prompt(method: str, text: str, cfg: Config, task: TaskConfig, worker: WorkerModel, seed: int) -> MethodResult:
    budget = BudgetManager(None)
    ev = Evaluator(worker, METRICS[task.evaluator], budget, cache=cfg.evaluation.cache)
    dev = ev.evaluate(text, task.dataset)
    return MethodResult(method=method, seed=seed, prompt_text=text, dev_accuracy=dev.accuracy, worker_calls=budget.worker_calls,
                        prompt_length=len(text))


def run_seed(cfg: Config, task: TaskConfig, worker: WorkerModel, seed: int) -> MethodResult:
    return evaluate_fixed_prompt("seed", serialize_prompt(parse_prompt(task.seed_prompt)), cfg, task, worker, seed)
