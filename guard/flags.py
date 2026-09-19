"""Unified JSONL flag/audit writer shared by every guardrail layer.

One JSON object per line. Every security-relevant event records which layer
flagged it (``layer``), the requesting user, the FULL masked prompt (never the
raw prompt), severity, reason, rules, layer-specific ``details``, and the
``audit_log`` row id when the event belongs to a unified chat request.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from guard.db import User

logger = logging.getLogger("guard.flags")

DEFAULT_FLAG_LOG = Path(__file__).resolve().parent.parent / "flags.jsonl"

EVENT_FLAG = "FLAG"
EVENT_RAG_QUERY = "RAG_QUERY"
EVENT_RAG_INJECTION = "RAG_CONTEXT_INJECTION"

LAYER_PROMPT_GUARD = "PROMPT_GUARD"
LAYER_MASKING = "MASKING"
LAYER_LLM_ROUTER = "LLM_ROUTER"
LAYER_ABAC = "ABAC"


def flag_log_path() -> Path:
    return Path(os.environ.get("GUARD_FLAG_LOG", DEFAULT_FLAG_LOG))


def append_flag(row: dict) -> None:
    """Append one JSON line; a caller-supplied ``ts`` wins over the default."""
    row = {"ts": datetime.now(timezone.utc).isoformat(), **row}
    path = flag_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("flag | %s row -> %s", row.get("event"), path)


def user_block(user: User | None) -> dict:
    return {"username": user.username, "role": user.role} if user else {
        "username": None,
        "role": None,
    }


def write_flag(
    *,
    layer: str | None,
    disposition: str,
    severity: str = "none",
    reason: str = "",
    rules=(),
    user: User | None = None,
    prompt_masked: str = "",
    details: dict | None = None,
    audit_id: int | None = None,
) -> None:
    """Append one unified FLAG row describing a guardrail violation."""
    append_flag(
        {
            "event": EVENT_FLAG,
            "layer": layer,
            "disposition": disposition,
            "severity": severity,
            "reason": reason,
            "rules": list(rules),
            "user": user_block(user),
            "prompt_masked": prompt_masked,
            "details": details or {},
            "audit_id": audit_id,
        }
    )
