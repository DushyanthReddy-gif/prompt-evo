"""Build worker/optimizer clients from config."""
from __future__ import annotations

import importlib

from prompt_evo.config import LLMSection
from prompt_evo.llm.base import LLMError, OptimizerModel, WorkerModel


def _load_class(path: str):
    mod, _, cls = path.partition(":")
    if not cls:
        raise LLMError("class_path must look like 'package.module:ClassName'")
    return getattr(importlib.import_module(mod), cls)


def build_worker(cfg: LLMSection) -> WorkerModel:
    if cfg.provider == "openai":
        from prompt_evo.llm.openai_compat import OpenAIWorker
        return OpenAIWorker(cfg)
    if cfg.provider == "custom" and cfg.class_path:
        w = _load_class(cfg.class_path)(**cfg.params)
        if not isinstance(w, WorkerModel):
            raise LLMError(f"{cfg.class_path} is not a WorkerModel")
        return w
    raise LLMError(f"unknown worker provider: {cfg.provider}")


def build_optimizer(cfg: LLMSection) -> OptimizerModel:
    if cfg.provider == "openai":
        from prompt_evo.llm.openai_compat import OpenAIOptimizer
        return OpenAIOptimizer(cfg)
    if cfg.provider == "custom" and cfg.class_path:
        o = _load_class(cfg.class_path)(**cfg.params)
        if not isinstance(o, OptimizerModel):
            raise LLMError(f"{cfg.class_path} is not an OptimizerModel")
        return o
    raise LLMError(f"unknown optimizer provider: {cfg.provider}")
