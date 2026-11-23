import pytest
from prompt_evo.operators import CrossoverOperator, FeedbackOperator, InvalidMutation, PRIMARY_OPERATORS, build_operators
from prompt_evo.prompts.parser import parse_prompt
from prompt_evo.core.budget import BudgetManager
from tests.conftest import SEED
from tests.fakes import ToyOptimizer


@pytest.mark.parametrize("name", list(PRIMARY_OPERATORS))
def test_each_primary_operator_changes_only_its_slot(name, ctx):
    parent = parse_prompt(SEED)
    for _ in range(6):  # a stochastic fake may occasionally no-op; retry
        try:
            out = PRIMARY_OPERATORS[name]().apply([parent], ctx)
            break
        except InvalidMutation:
            continue
    else:
        pytest.fail("operator never produced a valid child")
    op = PRIMARY_OPERATORS[name]()
    for slot, v in parent.slots_dict().items():
        if slot == op.slot or (slot == "instruction_ordering" and name == "instruction_wording"):
            continue
        assert out.child.slots_dict()[slot] == v, slot
    assert out.record.operator == name and out.record.original_component != out.record.new_component
    assert parent.constraints == []  # parent untouched


def test_exactly_six_primary_operators():
    assert len(PRIMARY_OPERATORS) == 6 and len(build_operators(list(PRIMARY_OPERATORS))) == 6
    with pytest.raises(ValueError):
        build_operators(["nope"])


def test_crossover_records_lineage(ctx):
    a = parse_prompt(SEED).model_copy(update={"version_id": "pa", "constraints": ["Only one label."]})
    b = a.model_copy(update={"version_id": "pb", "constraints": [], "output_format": "Answer: <answer>", "reasoning_instruction": "Verify."})
    for _ in range(8):
        try:
            out = CrossoverOperator().apply([a, b], ctx)
            break
        except InvalidMutation:
            continue
    assert out.record.parent_ids == ["pa", "pb"] and set(out.record.slot_sources) >= {"constraints", "output_format"}


@pytest.mark.parametrize("mode", ["empty", "garbage", "invalid"])
def test_invalid_llm_output_is_rejected_not_applied(mode, dev):
    import random
    from prompt_evo.operators import OperatorContext
    budget = BudgetManager(None)
    ctx = OperatorContext("t", ToyOptimizer(mode), budget, random.Random(0), pool=dev, max_retries=2)
    with pytest.raises(InvalidMutation):
        PRIMARY_OPERATORS["constraints"]().apply([parse_prompt(SEED)], ctx)
    assert budget.optimizer_calls == 3  # initial + 2 retries all counted


def test_retry_stops_when_optimizer_budget_exhausted(dev):
    import random
    from prompt_evo.operators import OperatorContext
    budget = BudgetManager(None, optimizer_budget=1)
    ctx = OperatorContext("t", ToyOptimizer("garbage"), budget, random.Random(0), pool=dev, max_retries=5)
    with pytest.raises(InvalidMutation):
        PRIMARY_OPERATORS["reasoning"]().apply([parse_prompt(SEED)], ctx)
    assert budget.optimizer_calls == 1


def test_output_format_must_mention_answer(ctx):
    op = PRIMARY_OPERATORS["output_format"]()
    with pytest.raises(ValueError):
        op.coerce("Respond in JSON", parse_prompt(SEED), ctx)


def test_feedback_operator_optional_and_works(ctx):
    ctx.failures = [{"input": "x", "expected": "y", "model_output": "z"}]
    out = FeedbackOperator().apply([parse_prompt(SEED)], ctx)
    assert out.child.constraints
