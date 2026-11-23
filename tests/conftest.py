import random
import pytest

from prompt_evo.config import Config, load_config
from prompt_evo.core.budget import BudgetManager
from prompt_evo.operators import OperatorContext
from prompt_evo.schemas.dataset import Example, TaskConfig, load_dataset
from tests.fakes import ToyOptimizer, ToyWorker

SEED = "Classify the sentiment of the review as positive or negative.\nUse the text of the review only.\nOutput one label."


@pytest.fixture
def dev():
    return load_dataset("data/classification_dev.jsonl")


@pytest.fixture
def task(dev):
    return TaskConfig(name="t", description="sentiment", evaluator="classification", dataset=dev,
                      test_dataset=load_dataset("data/classification_test.jsonl"), seed_prompt=SEED)


@pytest.fixture
def cfg():
    c = load_config("configs/example.yaml")
    c.optimizer.worker_budget, c.optimizer.population_size, c.optimizer.max_generations = 400, 4, 5
    return c


@pytest.fixture
def ctx(dev):
    return OperatorContext("sentiment", ToyOptimizer(), BudgetManager(None), random.Random(0), pool=dev)
