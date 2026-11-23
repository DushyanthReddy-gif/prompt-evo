"""Structured evolution event log (JSONL) mirrored to the stdlib logger."""
from __future__ import annotations

import json
import logging as std_logging
from pathlib import Path
from typing import Any

from prompt_evo.utils import now_iso

log = std_logging.getLogger("prompt_evo")


class EvolutionLogger:
    def __init__(self, path: str | Path | None = None) -> None:
        self.events: list[dict[str, Any]] = []
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        rec = {"event": event, "timestamp": now_iso(), **fields}
        self.events.append(rec)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        log.info("%s", json.dumps({k: v for k, v in rec.items() if k not in ("text", "timestamp")}, default=str)[:300])
        return rec

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(e, default=str) + "\n" for e in self.events))
