from prompt_evo.operators.base import InvalidMutation, MutationOutcome, Operator, OperatorContext, SlotOperator
from prompt_evo.operators.constraints import ConstraintsOperator
from prompt_evo.operators.crossover import CrossoverOperator
from prompt_evo.operators.feedback import FeedbackOperator
from prompt_evo.operators.few_shot import FewShotOperator
from prompt_evo.operators.instruction_ordering import InstructionOrderingOperator
from prompt_evo.operators.instruction_wording import InstructionWordingOperator
from prompt_evo.operators.output_format import OutputFormatOperator
from prompt_evo.operators.reasoning import ReasoningOperator

PRIMARY_OPERATORS: dict[str, type[SlotOperator]] = {
    "instruction_wording": InstructionWordingOperator, "instruction_ordering": InstructionOrderingOperator,
    "constraints": ConstraintsOperator, "few_shot": FewShotOperator,
    "output_format": OutputFormatOperator, "reasoning": ReasoningOperator,
}


def build_operators(names: list[str]) -> list[SlotOperator]:
    unknown = [n for n in names if n not in PRIMARY_OPERATORS]
    if unknown:
        raise ValueError(f"unknown operators: {unknown}; valid: {list(PRIMARY_OPERATORS)}")
    return [PRIMARY_OPERATORS[n]() for n in names]


__all__ = ["InvalidMutation", "MutationOutcome", "Operator", "OperatorContext", "SlotOperator", "PRIMARY_OPERATORS",
           "build_operators", "CrossoverOperator", "FeedbackOperator"]
