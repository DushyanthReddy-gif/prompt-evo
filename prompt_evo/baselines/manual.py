"""Baseline 2: user-provided manual prompt. Never invented: unavailable if not provided."""
from __future__ import annotations

from prompt_evo.baselines.seed import evaluate_fixed_prompt
from prompt_evo.config import Config
from prompt_evo.llm.base import WorkerModel
from prompt_evo.prompts.parser import parse_prompt
from prompt_evo.prompts.serializer import serialize_prompt
from prompt_evo.schemas.dataset import TaskConfig
from prompt_evo.schemas.evolution import MethodResult


def run_manual(cfg: Config, task: TaskConfig, worker: WorkerModel, seed: int) -> MethodResult:
    if not task.manual_prompt:
        return MethodResult(method="manual", seed=seed, available=False, note="no manual prompt provided; baseline unavailable")
    return evaluate_fixed_prompt("manual", serialize_prompt(parse_prompt(task.manual_prompt)), cfg, task, worker, seed)
