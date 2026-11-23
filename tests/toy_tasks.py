"""Toy task generators + oracles (TEST/TOY ONLY, used by the deterministic fake worker)."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

POS = ["great", "excellent", "wonderful", "fantastic", "delightful", "superb"]
NEG = ["terrible", "awful", "dreadful", "horrible", "disappointing", "poor"]
NOUNS = ["movie", "meal", "service", "book", "hotel", "game", "show", "phone"]
CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Spain": "Madrid", "Germany": "Berlin", "Egypt": "Cairo",
            "Canada": "Ottawa", "Brazil": "Brasilia", "India": "New Delhi", "China": "Beijing", "Russia": "Moscow",
            "Kenya": "Nairobi", "Peru": "Lima", "Chile": "Santiago", "Norway": "Oslo", "Sweden": "Stockholm",
            "Greece": "Athens", "Turkey": "Ankara", "Cuba": "Havana", "Ireland": "Dublin", "Poland": "Warsaw",
            "Austria": "Vienna", "Portugal": "Lisbon", "Thailand": "Bangkok", "Vietnam": "Hanoi", "Ghana": "Accra",
            "Nepal": "Kathmandu", "Iran": "Tehran", "Iraq": "Baghdad", "Cuba2": "Havana"}
QA_TEMPLATES = ["Q: What is the capital of {c}?", "Q: Name the capital city of {c}.", "Q: {c}'s capital is which city?", "Q: Which city is the capital of {c}?"]


def truth(inp: str) -> str:
    m = re.search(r"What is (-?\d+) ([+\-*]) (-?\d+)\?", inp)
    if m:
        a, op, b = int(m[1]), m[2], int(m[3])
        return str(a + b if op == "+" else a - b if op == "-" else a * b)
    if inp.startswith("Review:"):
        words = re.findall(r"[a-z]+", inp.lower())
        pos = any(w in POS for w in words)
        neg_adj = any(w in NEG for w in words)
        flipped = "not" in words
        return "positive" if (pos ^ flipped) and not neg_adj or (neg_adj and flipped) else "negative"
    for c, cap in CAPITALS.items():
        if c.rstrip("2") in inp:
            return cap
    return "unknown"


def wrong(inp: str, tr: str) -> str:
    if tr in ("positive", "negative"):
        return "negative" if tr == "positive" else "positive"
    if re.fullmatch(r"-?\d+", tr):
        return str(int(tr) + 1)
    return "Unknown"


def gen_classification(n: int, rng: random.Random) -> list[dict]:
    out = []
    for i in range(n):
        adj, neg = rng.choice(POS + NEG), rng.random() < 0.3
        s = f"Review: The {rng.choice(NOUNS)} was {'not ' if neg else ''}{adj}."
        out.append({"id": f"c{i:03d}", "input": s, "expected_output": truth(s)})
    return out


def gen_math(n: int, rng: random.Random) -> list[dict]:
    out = []
    for i in range(n):
        s = f"What is {rng.randint(2, 40)} {rng.choice('+-*')} {rng.randint(2, 20)}?"
        out.append({"id": f"m{i:03d}", "input": s, "expected_output": truth(s)})
    return out


def gen_qa(n: int, rng: random.Random) -> list[dict]:
    cs = [c for c in CAPITALS if c != "Cuba2"]
    out = []
    for i in range(n):
        c = rng.choice(cs)
        out.append({"id": f"q{i:03d}", "input": rng.choice(QA_TEMPLATES).format(c=c), "expected_output": CAPITALS[c]})
    return out


def write_all(root: str | Path = "data", seed: int = 0) -> None:
    root = Path(root)
    root.mkdir(exist_ok=True)
    rng = random.Random(seed)
    for name, gen in (("classification", gen_classification), ("math", gen_math), ("qa", gen_qa)):
        for split in ("dev", "test"):
            rows = gen(60, rng)
            (root / f"{name}_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
