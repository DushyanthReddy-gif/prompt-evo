"""Dataset and task schemas plus JSON/JSONL/CSV loaders."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field


class DatasetError(ValueError):
    """Raised for invalid or unreadable datasets."""


class Example(BaseModel):
    id: str
    input: str
    expected_output: str


class TaskConfig(BaseModel):
    """Runtime task: description, data, evaluator name and prompts."""

    name: str
    description: str
    evaluator: str
    dataset: list[Example]
    seed_prompt: str
    test_dataset: list[Example] = Field(default_factory=list)
    fewshot_pool: list[Example] = Field(default_factory=list)
    manual_prompt: str | None = None


def _validate(rows: list[dict], path: str) -> list[Example]:
    if not rows:
        raise DatasetError(f"{path}: dataset is empty")
    out: list[Example] = []
    seen: set[str] = set()
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or "input" not in r or "expected_output" not in r:
            raise DatasetError(f"{path}: row {i} needs 'input' and 'expected_output'")
        ex = Example(id=str(r.get("id") or f"ex{i:03d}"), input=str(r["input"]), expected_output=str(r["expected_output"]))
        if not ex.input.strip() or not ex.expected_output.strip():
            raise DatasetError(f"{path}: row {i} has empty input/expected_output")
        if ex.id in seen:
            raise DatasetError(f"{path}: duplicate id {ex.id}")
        seen.add(ex.id)
        out.append(ex)
    return out


def load_dataset(path: str | Path) -> list[Example]:
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"dataset not found: {p}")
    try:
        if p.suffix == ".jsonl":
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        elif p.suffix == ".json":
            rows = json.loads(p.read_text())
        elif p.suffix == ".csv":
            with p.open(newline="") as f:
                rows = list(csv.DictReader(f))
        else:
            raise DatasetError(f"unsupported dataset format: {p.suffix}")
    except (json.JSONDecodeError, csv.Error) as e:
        raise DatasetError(f"{p}: cannot parse ({e})") from e
    if not isinstance(rows, list):
        raise DatasetError(f"{p}: expected a list of examples")
    return _validate(rows, str(p))


def dataset_hash(examples: list[Example]) -> str:
    payload = json.dumps([e.model_dump() for e in examples], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
