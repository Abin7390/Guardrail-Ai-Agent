"""Guard pipeline: jailbreak check first, Presidio masking second, flag logging."""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from guard.steps.masking import MaskingResult, mask
from guard.steps.prompt_guard import PromptGuardVerdict, classify, get_threshold

logger = logging.getLogger("guard")

RULE_JAILBREAK = "PROMPT_GUARD_SUSPICIOUS"
RULE_PII = "PII_DETECTED"
SNIPPET_LIMIT = 120

DEFAULT_FLAG_LOG = Path(__file__).resolve().parent.parent / "flags.jsonl"


@dataclass(frozen=True)
class GuardResult:
    disposition: str
    flagged: bool
    rules: list[str]
    verdict: PromptGuardVerdict | None
    masking: MaskingResult | None
    masked_prompt: str | None

def _flag_log_path() -> Path:
    return Path(os.environ.get("GUARD_FLAG_LOG", DEFAULT_FLAG_LOG))


def write_flag(result: GuardResult, raw: str) -> None:
    """Append one JSON line for a flagged event; never stores raw PII."""
    if result.masking is not None:
        masked = result.masking.masked_text
    else:
        masked = mask(raw).masked_text
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "disposition": result.disposition,
        "rules": result.rules,
        "label": result.verdict.label if result.verdict else None,
        "suspicious_score": result.verdict.suspicious_score if result.verdict else None,
        "matched": result.verdict.matched if result.verdict else None,
        "snippet": masked[:SNIPPET_LIMIT],
    }
    path = _flag_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("flag logged -> %s", path)


def screen(raw: str, reversible: bool = False) -> GuardResult:
    """Screen one raw prompt: REJECT on jailbreak, else MASKED/CLEAN after Presidio.

    ``reversible=True`` masks with indexed ``[REDACTED_n]`` placeholders and
    carries the value mapping on ``GuardResult.masking`` for later demasking;
    the flag-log snippet behavior is unchanged either way.
    """
    logger.info("step 1/2 | jailbreak check: running Prompt-Guard classifier")
    verdict = classify(raw)
    threshold = get_threshold()
    if verdict.flagged:
        detail = (
            f" | supplemental match: {verdict.matched}" if verdict.matched else ""
        )
        logger.info(
            "step 1/2 | RESULT: %s score=%.4f >= threshold %.2f (engine=%s)%s -> REJECT, pipeline halted",
            verdict.label,
            verdict.suspicious_score,
            threshold,
            verdict.engine,
            detail,
        )
        result = GuardResult(
            disposition="REJECT",
            flagged=True,
            rules=[RULE_JAILBREAK],
            verdict=verdict,
            masking=None,
            masked_prompt=None,
        )
        write_flag(result, raw)
        return result
    logger.info(
        "step 1/2 | RESULT: %s score=%.4f < threshold %.2f (engine=%s) -> proceed to masking",
        verdict.label,
        verdict.suspicious_score,
        threshold,
        verdict.engine,
    )
    logger.info("step 2/2 | PII masking: scanning with Presidio")
    masking_result = mask(raw, reversible=True) if reversible else mask(raw)
    if masking_result.entities:
        logger.info(
            "step 2/2 | RESULT: PII found %s (engine=%s) -> MASKED",
            masking_result.entities,
            masking_result.engine,
        )
        rules = [RULE_PII] + [
            f"PII_{entity}" for entity in sorted(masking_result.entities)
        ]
        result = GuardResult(
            disposition="MASKED",
            # disposition="REJECT",
            flagged=True,
            rules=rules,
            verdict=verdict,
            masking=masking_result,
            masked_prompt=masking_result.masked_text,
        )
        write_flag(result, raw)
        return result
    logger.info(
        "step 2/2 | RESULT: no PII found (engine=%s) -> CLEAN", masking_result.engine
    )
    return GuardResult(
        disposition="CLEAN",
        flagged=False,
        rules=[],
        verdict=verdict,
        masking=masking_result,
        masked_prompt=raw,
    )


