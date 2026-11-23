"""YAML configuration (pydantic-validated)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

ALL_OPERATORS = [
    "instruction_wording", "instruction_ordering", "constraints",
    "few_shot", "output_format", "reasoning",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMSection(_Strict):
    """Used for both the frozen WORKER and the OPTIMIZER (mutation) model."""
    provider: str = "openai"          # 'openai' (any OpenAI-compatible endpoint) | 'custom'
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    max_retries: int = 5
    timeout_s: float = 60.0
    class_path: str | None = None     # for provider=custom: 'package.module:ClassName'
    params: dict[str, Any] = Field(default_factory=dict)
    price_per_1k_prompt: float = 0.0
    price_per_1k_completion: float = 0.0


class TaskSection(_Strict):
    name: str = "task"
    description: str = ""
    type: str = "classification"       # classification | math | qa (selects evaluator)
    dataset: str = ""                  # dev set path
    test_dataset: str | None = None
    fewshot_pool: str | None = None    # optional separate pool (avoids dev leakage)
    seed_prompt: str = ""              # file path or inline text
    manual_prompt: str | None = None   # file path or inline text


class RacingSection(_Strict):
    enabled: bool = True
    min_examples: int = 10
    margin: float = 0.0
    use_bound: bool = True             # stop when max achievable score < reference
    use_confidence: bool = False       # stop when Wilson upper bound < reference - margin
    z: float = 1.96
    reference: str = "parent"          # parent | best


class EvaluationSection(_Strict):
    initial_slice_size: int = 20
    promotion_slice_size: int = 50
    promote_tolerance: float = 0.0
    bootstrap_samples: int = 100
    length_penalty: float = 0.01
    max_prompt_chars: int = 2000       # normalizer for prompt length
    hard_max_prompt_chars: int = 6000
    cache: bool = True                 # valid when worker temperature == 0
    eval_timeout_s: float | None = None
    racing: RacingSection = Field(default_factory=RacingSection)


class OptimizerSection(_Strict):
    population_size: int = 8
    max_generations: int = 20
    worker_budget: int = 1000
    max_optimizer_calls: int | None = None
    mutation_probability: float = 0.8
    crossover_probability: float = 0.2
    feedback_probability: float = 0.0  # optional feedback operator (off by default)
    seed: int = 42
    operators: list[str] = Field(default_factory=lambda: list(ALL_OPERATORS))
    operator_weights: dict[str, float] = Field(default_factory=dict)
    selection: str = "tournament"      # tournament | pareto
    tournament_size: int = 3
    robust_penalty: float = 0.5        # score = fitness - robust_penalty * bootstrap_std
    adaptive: bool = False             # Thompson-sampling operator selection (stretch)
    max_retries: int = 1
    max_fewshot: int = 6
    init_attempt_factor: int = 4
    children_per_generation: int | None = None


class ExperimentSection(_Strict):
    name: str = "experiment"
    output_dir: str = "results"
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3])
    budgets: list[int] = Field(default_factory=lambda: [500, 1000, 2000])
    methods: list[str] = Field(default_factory=lambda: ["seed", "manual", "genetic", "promptevo"])
    extra_workers: list[LLMSection] = Field(default_factory=list)
    extra_tasks: list[TaskSection] = Field(default_factory=list)
    transfer_worker: LLMSection | None = None


class Config(_Strict):
    task: TaskSection = Field(default_factory=TaskSection)
    optimizer: OptimizerSection = Field(default_factory=OptimizerSection)
    evaluation: EvaluationSection = Field(default_factory=EvaluationSection)
    worker: LLMSection = Field(default_factory=LLMSection)
    optimizer_llm: LLMSection = Field(default_factory=LLMSection)
    experiment: ExperimentSection = Field(default_factory=ExperimentSection)


def load_config(path: str | Path) -> Config:
    p = Path(path)
    data = yaml.safe_load(p.read_text()) or {}
    return Config.model_validate(data)


def resolve_text(value: str | None, base: Path | None = None) -> str | None:
    """Return file contents if ``value`` is an existing path, else the literal text."""
    if value is None:
        return None
    for cand in [Path(value)] + ([base / value] if base else []):
        try:
            if cand.is_file():
                return cand.read_text().strip()
        except OSError:
            pass
    return value.strip()
