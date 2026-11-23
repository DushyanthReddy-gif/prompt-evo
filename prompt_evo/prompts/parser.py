"""Raw text / explicit JSON-YAML -> StructuredPrompt.

Automatic parsing is a best-effort heuristic on section headers; explicit structured
prompts (JSON/YAML with the six slot names) bypass it entirely.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from prompt_evo.schemas.prompt import SLOTS, FewShotExample, StructuredPrompt

_HEADER = re.compile(r"^\s*(constraints|reasoning|output format|examples)\s*:\s*(.*)$", re.I)
_KEY = {"constraints": "constraints", "reasoning": "reasoning_instruction",
        "output format": "output_format", "examples": "few_shot_examples"}


class PromptParseError(ValueError):
    pass


def _parse_examples(lines: list[str]) -> list[FewShotExample]:
    text = "\n".join(lines).strip()
    out = []
    for block in re.split(r"\n\s*\n", text):
        m = re.match(r"\s*Input:\s*(.*?)\n\s*Output:\s*(.*)$", block.strip(), re.S | re.I)
        if m:
            out.append(FewShotExample(input=m.group(1), output=m.group(2)))
    return out


def _explicit(text: str) -> dict | None:
    try:
        if text.startswith("{"):
            d = json.loads(text)
        elif "instruction_wording" in text:
            d = yaml.safe_load(text)
        else:
            return None
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise PromptParseError(f"invalid structured prompt: {e}") from e
    if not isinstance(d, dict) or "instruction_wording" not in d:
        return None
    if isinstance(d["instruction_wording"], str):
        d["instruction_wording"] = [ln for ln in d["instruction_wording"].splitlines() if ln.strip()]
    return {k: v for k, v in d.items() if k in SLOTS}


def parse_prompt(raw: str, version_id: str = "p0") -> StructuredPrompt:
    """Parse ``raw`` into a StructuredPrompt. Never loses text: unrecognised content
    stays in ``instruction_wording``."""
    text = raw.strip()
    if not text:
        raise PromptParseError("empty prompt")
    slots = _explicit(text)
    if slots is not None:
        return StructuredPrompt(**slots, raw_prompt=raw, version_id=version_id)
    sections: dict[str, list[str]] = {"instructions": []}
    cur = "instructions"
    for line in text.splitlines():
        m = _HEADER.match(line)
        if m:
            cur = _KEY[m.group(1).lower()]
            sections.setdefault(cur, [])
            if m.group(2).strip():
                sections[cur].append(m.group(2).strip())
            continue
        sections[cur].append(line)
    instr = [ln.strip() for ln in sections["instructions"] if ln.strip()]
    if not instr:
        raise PromptParseError("no instruction text found")
    cons = [re.sub(r"^[-*•]\s*", "", ln.strip()) for ln in sections.get("constraints", []) if ln.strip()]
    return StructuredPrompt(
        instruction_wording=instr, constraints=cons,
        few_shot_examples=_parse_examples(sections.get("few_shot_examples", [])),
        output_format=" ".join(ln.strip() for ln in sections.get("output_format", [])),
        reasoning_instruction=" ".join(ln.strip() for ln in sections.get("reasoning_instruction", [])),
        raw_prompt=raw, version_id=version_id,
    )


def load_structured_prompt_file(path: str | Path) -> StructuredPrompt:
    return parse_prompt(Path(path).read_text())


def check_compatible(p: StructuredPrompt, max_chars: int = 6000) -> None:
    """Raise ValueError if a prompt would break the evaluation harness or is degenerate."""
    from prompt_evo.prompts.serializer import serialize_prompt
    if p.output_format and "answer" not in p.output_format.lower():
        raise ValueError("output_format must mention 'answer' so the harness can extract it")
    if len(serialize_prompt(p)) > max_chars:
        raise ValueError("prompt too long")
