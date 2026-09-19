"""Guard pipeline: jailbreak check first, Presidio masking second, flag logging.

Flag rows are written through :mod:`guard.flags` using the unified schema
(layer, user, full masked prompt, severity, reason, rules, details).
"""

import logging
from dataclasses import dataclass

from guard.db import User
from guard.flags import (
    LAYER_MASKING,
    LAYER_PROMPT_GUARD,
    write_flag,
)
from guard.steps.masking import MaskingResult, mask
from guard.steps.prompt_guard import PromptGuardVerdict, classify, get_threshold

logger = logging.getLogger("guard")

RULE_JAILBREAK = "PROMPT_GUARD_SUSPICIOUS"
RULE_PII = "PII_DETECTED"


@dataclass(frozen=True)
class GuardResult:
    disposition: str
    flagged: bool
    rules: list[str]
    verdict: PromptGuardVerdict | None
    masking: MaskingResult | None
    masked_prompt: str | None


def screen(raw: str, reversible: bool = False, user: User | None = None) -> GuardResult:
    """Screen one raw prompt: REJECT on jailbreak, else MASKED/CLEAN after Presidio.

    ``reversible=True`` masks with indexed ``[REDACTED_n]`` placeholders and
    carries the value mapping on ``GuardResult.masking`` for later demasking;
    the flag rows always store the masked text, never the raw prompt. When
    ``user`` is provided it is recorded on every flag row written here.
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
        write_flag(
            layer=LAYER_PROMPT_GUARD,
            disposition="REJECT",
            severity="high",
            reason="jailbreak or prompt injection detected",
            rules=result.rules,
            user=user,
            prompt_masked=mask(raw).masked_text,
            details={
                "label": verdict.label,
                "suspicious_score": verdict.suspicious_score,
                "threshold": threshold,
                "engine": verdict.engine,
                "matched": verdict.matched,
            },
        )
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
            flagged=True,
            rules=rules,
            verdict=verdict,
            masking=masking_result,
            masked_prompt=masking_result.masked_text,
        )
        write_flag(
            layer=LAYER_MASKING,
            disposition="MASKED",
            severity="low",
            reason="personal data detected and masked",
            rules=rules,
            user=user,
            prompt_masked=masking_result.masked_text,
            details={
                "entities": masking_result.entities,
                "engine": masking_result.engine,
            },
        )
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
