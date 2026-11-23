import pytest
from prompt_evo.prompts.parser import PromptParseError, parse_prompt
from prompt_evo.prompts.serializer import prompt_hash, serialize_prompt
from prompt_evo.schemas.prompt import StructuredPrompt

FULL = """Classify the review.
Be accurate.

Constraints:
- Only one label.
- No explanations.

Reasoning: Check your work.
Output format: Finish with 'Answer: <answer>'.

Examples:
Input: Review: great
Output: positive

Input: Review: awful
Output: negative"""


def test_parse_headers():
    p = parse_prompt(FULL)
    assert p.instruction_wording == ["Classify the review.", "Be accurate."]
    assert p.constraints == ["Only one label.", "No explanations."]
    assert p.reasoning_instruction == "Check your work."
    assert "Answer" in p.output_format and len(p.few_shot_examples) == 2


def test_plain_prompt_parses_to_instructions():
    p = parse_prompt("Just do the task")
    assert p.instruction_wording == ["Just do the task"] and p.constraints == []


def test_serialize_roundtrip_idempotent():
    s1 = serialize_prompt(parse_prompt(FULL))
    assert serialize_prompt(parse_prompt(s1)) == s1


def test_explicit_json_and_yaml():
    j = parse_prompt('{"instruction_wording": ["A", "B"], "constraints": ["c"], "instruction_ordering": [1, 0]}')
    assert serialize_prompt(j).startswith("B\nA")
    y = parse_prompt("instruction_wording:\n  - X\noutput_format: 'Answer: <answer>'")
    assert y.output_format == "Answer: <answer>"


def test_empty_and_bad_ordering():
    with pytest.raises(PromptParseError):
        parse_prompt("   ")
    with pytest.raises(ValueError):
        StructuredPrompt(instruction_wording=["a", "b"], instruction_ordering=[0, 0])


def test_duplicate_detection_hash_ignores_case_whitespace():
    a = parse_prompt("Do  the TASK")
    b = parse_prompt("do the task")
    assert prompt_hash(a) == prompt_hash(b)
