"""Deterministic fakes for TESTS and the offline toy experiment ONLY (never used by production paths
unless explicitly referenced via provider=custom, class_path=tests.fakes:...)."""
from __future__ import annotations

import json
import random
import re

from prompt_evo.llm.base import LLMResponse, OptimizerModel, WorkerModel
from prompt_evo.utils import extract_json, stable_hash
from tests.toy_tasks import truth, wrong

GOOD_CONSTRAINTS = ["Answer with only the final answer, no extra words.", "Do not add explanations after the answer.",
                    "If uncertain, choose the most likely answer.", "Read the input carefully before answering.",
                    "Never leave the answer empty."]
FORMATS = ["Finish with a line 'Answer: <answer>'.", "Return the answer on one line as 'Answer: <answer>'.",
           "Give a concise answer.", "Reply with JSON {\"answer\": \"<answer>\"}."]
REASONING = ["Verify your answer before responding.", "Break the problem into steps and double-check the result.",
             "Think briefly, then check your work.", "Reason internally and report only the answer."]
WORDING = ["Be precise.", "Read the input closely.", "Respond accurately."]


class ToyWorker(WorkerModel):
    """Deterministic simulated worker: P(correct) depends on prompt features (constraints, format, reasoning,
    few-shot) with per-example noise that is mostly prompt-independent so improvements are paired."""
    temperature = 0.0

    def __init__(self, name: str = "toy-worker", skill: float = 0.0) -> None:
        self.name, self.skill = name, skill

    def p_correct(self, prompt: str) -> float:
        low, p = prompt.lower(), 0.52 + self.skill
        p += 0.10 * ("only the final answer" in low) + 0.05 * ("do not add explanations" in low)
        p += 0.03 * ("most likely" in low) + 0.02 * ("carefully" in low)
        p += 0.06 * ("answer: <answer>" in low) + 0.05 * ("verify" in low or "double-check" in low or "check your work" in low)
        p += 0.03 * min(prompt.count("Input:"), 3)
        p += 0.02 * ("be precise" in low)
        return min(p, 0.97)

    def generate(self, prompt: str, input: str) -> LLMResponse:
        u_ex = (stable_hash("ex", input) % 10_000) / 10_000
        u_pr = (stable_hash("pr", prompt, input) % 10_000) / 10_000
        ok = (0.85 * u_ex + 0.15 * u_pr) < self.p_correct(prompt)
        tr = truth(input)
        ans = tr if ok else wrong(input, tr)
        text = f"Answer: {ans}" if "answer: <answer>" in prompt.lower() else ans
        return LLMResponse(text=text, prompt_tokens=len(prompt) // 4, completion_tokens=3, call_id=None)


class ToyOptimizer(OptimizerModel):
    """Deterministic simulated optimizer LLM: understands the operator templates and returns valid JSON."""
    name = "toy-optimizer"

    def __init__(self, mode: str = "ok") -> None:
        self.mode, self.calls = mode, 0

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls += 1
        if self.mode == "empty":
            return LLMResponse(text="")
        if self.mode == "garbage":
            return LLMResponse(text="I think you should improve the prompt!")
        if self.mode == "invalid":
            return LLMResponse(text=json.dumps({"new_component": 12345, "reason": "bad"}))
        rng = random.Random(stable_hash(user, self.calls))
        op = re.search(r"OPERATOR: (\w+)", user).group(1)
        return LLMResponse(text=json.dumps(getattr(self, f"_{op}")(user, rng)), call_id=f"toy-{self.calls}")

    @staticmethod
    def _current(user: str):
        return json.loads(re.search(r"CURRENT_COMPONENT_JSON: (.*)", user).group(1))

    def _instruction_wording(self, user, rng):
        cur = self._current(user)
        cur = list(cur)
        cur.append(rng.choice(WORDING)) if rng.random() < 0.5 else cur.__setitem__(rng.randrange(len(cur)), cur[0] + " " + rng.choice(WORDING))
        return {"new_component": cur, "reason": "tightened wording"}

    def _instruction_ordering(self, user, rng):
        cur = self._current(user)
        new = cur[:]
        rng.shuffle(new)
        return {"new_component": new, "reason": "reordered"}

    def _constraints(self, user, rng):
        cur = self._current(user)
        if cur and rng.random() < 0.2:
            cur.pop(rng.randrange(len(cur)))
        else:
            cur.append(rng.choice(GOOD_CONSTRAINTS))
        return {"new_component": list(dict.fromkeys(cur)), "reason": "adjusted constraints"}

    def _few_shot(self, user, rng):
        ids = re.findall(r"^\[(\w+)\]", user, re.M)
        cur = self._current(user)
        if ids and len(cur) < 6:
            cur = cur + [{"id": rng.choice(ids)}]
        elif cur:
            cur = cur[1:]
        return {"new_component": cur, "reason": "changed examples"}

    def _output_format(self, user, rng):
        return {"new_component": rng.choice(FORMATS), "reason": "format change"}

    def _reasoning(self, user, rng):
        return {"new_component": rng.choice(REASONING), "reason": "reasoning change"}

    def _crossover(self, user, rng):
        a = json.loads(re.search(r"PARENT_A_JSON: (.*)", user).group(1))
        b = json.loads(re.search(r"PARENT_B_JSON: (.*)", user).group(1))
        child = {s: rng.choice([a, b])[s] for s in a}
        child["instruction_ordering"] = list(range(len(child["instruction_wording"])))
        return {"child": child, "reason": "combined parents"}

    def _feedback(self, user, rng):
        p = json.loads(re.search(r"PARENT_JSON: (.*)", user).group(1))
        p["constraints"] = list(dict.fromkeys(p["constraints"] + [rng.choice(GOOD_CONSTRAINTS)]))
        return {"child": p, "reason": "targeted failures"}

    def _generic_mutation(self, user, rng):
        t = re.search(r"PROMPT:\n<<<\n(.*?)\n>>>", user, re.S).group(1)
        add = rng.choice(GOOD_CONSTRAINTS + REASONING + FORMATS + WORDING)
        return {"prompt": t + "\n" + add, "reason": "appended"}

    def _generic_crossover(self, user, rng):
        a = re.search(r"PROMPT_A:\n<<<\n(.*?)\n>>>", user, re.S).group(1).splitlines()
        b = re.search(r"PROMPT_B:\n<<<\n(.*?)\n>>>", user, re.S).group(1).splitlines()
        lines = list(dict.fromkeys(a[: len(a) // 2 + 1] + b[len(b) // 2 :]))
        return {"prompt": "\n".join(lines), "reason": "crossed"}
