"""Answer extraction and task-specific correctness metrics (deterministic)."""
from __future__ import annotations

import json
import re
from typing import Callable, Sequence

Metric = Callable[[str, str], bool]  # (raw_model_output, expected) -> correct


def normalize_text(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_answer(text: str) -> tuple[str, bool]:
    """Return (answer, explicit). Looks for JSON {"answer": ...} or an 'Answer: ...' line;
    otherwise falls back to the last non-empty line (explicit=False)."""
    t = text.strip()
    if t.startswith("{"):
        try:
            d = json.loads(t)
            if isinstance(d, dict) and "answer" in d:
                return str(d["answer"]).strip(), True
        except json.JSONDecodeError:
            pass
    matches = re.findall(r"(?im)^\W*(?:final\s+)?answer\s*[:=\-]\s*(.+?)\s*$", t)
    if matches:
        return matches[-1].strip(), True
    lines = [ln for ln in t.splitlines() if ln.strip()]
    return (lines[-1].strip() if lines else ""), False


def _to_number(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").replace("%", "")
    m = re.findall(r"-?\d+\s*/\s*\d+|-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    tok = m[-1].replace(" ", "")
    if "/" in tok:
        n, d = tok.split("/")
        return float(n) / float(d) if float(d) != 0 else None
    return float(tok)


def classification_match(output: str, expected: str) -> bool:
    return normalize_text(extract_answer(output)[0]) == normalize_text(expected)


def math_match(output: str, expected: str) -> bool:
    got, want = _to_number(extract_answer(output)[0]), _to_number(expected)
    if want is None:
        return normalize_text(extract_answer(output)[0]) == normalize_text(expected)
    return got is not None and abs(got - want) <= 1e-6 * max(1.0, abs(want))


def qa_match(output: str, expected: str) -> bool:
    pred, gold = normalize_text(extract_answer(output)[0]).split(), normalize_text(expected).split()
    if not gold:
        return False
    if pred == gold:
        return True
    if len(pred) > 3 * len(gold) + 3:
        return False
    return any(pred[i : i + len(gold)] == gold for i in range(len(pred) - len(gold) + 1))


METRICS: dict[str, Metric] = {
    "classification": classification_match, "math": math_match, "qa": qa_match, "exact": classification_match,
}


def accuracy(flags: Sequence[bool]) -> float:
    return sum(map(bool, flags)) / len(flags) if flags else 0.0
