"""Small shared helpers (hashing, JSON extraction, metadata)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
from typing import Any


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def stable_hash(*parts: Any) -> int:
    """Deterministic integer hash (Python's hash() is salted per process)."""
    return int(hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM reply (handles code fences and chatter)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start = t.find("{")
        if start < 0:
            raise ValueError("no JSON object found in reply")
        depth, in_str, esc, obj = 0, False, False, None
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    obj = json.loads(t[start : i + 1])
                    break
        if obj is None:
            raise ValueError("unbalanced JSON in reply")
    if not isinstance(obj, dict):
        raise ValueError("JSON reply is not an object")
    return obj
