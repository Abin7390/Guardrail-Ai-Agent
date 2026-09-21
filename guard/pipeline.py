"""Guard pipeline: jailbreak check first, Presidio masking second, flag logging.

Flag rows are written through :mod:`guard.flags` using the unified schema
(layer, user, full masked prompt, severity, reason, rules, details).
"""

import logging
import re
from dataclasses import dataclass

from guard.db import User
from guard.flags import (
    LAYER_MASKING,
    LAYER_PROMPT_GUARD,
    write_flag,
)
from guard.steps.file_intake import (
    ExtractedFile,
    assemble_with_files,
    split_masked,
)
from guard.steps.masking import MaskedEntity, MaskingResult, mask
from guard.steps.prompt_guard import PromptGuardVerdict, classify, get_threshold

logger = logging.getLogger("guard")

RULE_JAILBREAK = "PROMPT_GUARD_SUSPICIOUS"
RULE_PII = "PII_DETECTED"
RULE_FILE_INJECTION = "FILE_PROMPT_GUARD_SUSPICIOUS"

_PLACEHOLDER_SCAN = re.compile(r"\[REDACTED_\d+\]")


@dataclass(frozen=True)
class GuardResult:
    disposition: str
    flagged: bool
    rules: list[str]
    verdict: PromptGuardVerdict | None
    masking: MaskingResult | None
    masked_prompt: str | None
    files: list[dict] | None = None
    masked_files: list[tuple[str, str]] | None = None


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


def _section_entities(
    mapping: list[MaskedEntity] | None, section_text: str
) -> dict[str, int]:
    """Count masked entities per text section using the reversible mapping."""
    if not mapping:
        return {}
    by_placeholder = {entity.placeholder: entity.entity_type for entity in mapping}
    counts: dict[str, int] = {}
    for token in _PLACEHOLDER_SCAN.findall(section_text):
        entity_type = by_placeholder.get(token)
        if entity_type:
            counts[entity_type] = counts.get(entity_type, 0) + 1
    return counts


def _files_report(
    files: list[ExtractedFile],
    file_verdicts: list[PromptGuardVerdict],
    section_entities: dict[int, dict[str, int]] | None,
) -> list[dict]:
    """Per-file audit/flag metadata; verdicts absent when the pipeline halted early."""
    sections = section_entities or {}
    report = []
    for index, extracted in enumerate(files):
        verdict = file_verdicts[index] if index < len(file_verdicts) else None
        report.append(
            {
                "filename": extracted.filename,
                "extension": extracted.extension,
                "size_bytes": extracted.size_bytes,
                "label": verdict.label if verdict else None,
                "suspicious_score": verdict.suspicious_score if verdict else None,
                "engine": verdict.engine if verdict else None,
                "entities": sections.get(index, {}),
            }
        )
    return report


def screen_request(
    prompt: str,
    files: list[ExtractedFile] | None = None,
    *,
    reversible: bool = False,
    user: User | None = None,
) -> GuardResult:
    """Screen one prompt plus its attached files through every guard step.

    Same semantics as :func:`screen`, generalized for uploads: the prompt and
    every extracted file text are classified separately (a jailbreak hidden in
    a file rejects the whole request even when the prompt is benign), then the
    surviving texts are masked as ONE combined document so reversible
    placeholders stay unique across prompt and files, and the result is split
    back apart. ``files`` carries the per-file report (audit/flag metadata,
    never file text) and ``masked_files`` the ``(filename, masked_text)``
    pairs the router and answer LLM see.
    """
    files = list(files or [])
    if not files:
        return screen(prompt, reversible=reversible, user=user)

    logger.info(
        "step 1/2 | jailbreak check: prompt + %d attached file(s)", len(files)
    )
    threshold = get_threshold()
    verdict = classify(prompt)
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
            files=_files_report(files, [], None),
        )
        write_flag(
            layer=LAYER_PROMPT_GUARD,
            disposition="REJECT",
            severity="high",
            reason="jailbreak or prompt injection detected",
            rules=result.rules,
            user=user,
            prompt_masked=mask(prompt).masked_text,
            details={
                "label": verdict.label,
                "suspicious_score": verdict.suspicious_score,
                "threshold": threshold,
                "engine": verdict.engine,
                "matched": verdict.matched,
            },
        )
        return result

    file_verdicts: list[PromptGuardVerdict] = []
    for extracted in files:
        file_verdict = classify(extracted.text)
        file_verdicts.append(file_verdict)
        if file_verdict.flagged:
            file_detail = (
                f" | supplemental match: {file_verdict.matched}"
                if file_verdict.matched
                else ""
            )
            logger.info(
                "step 1/2 | RESULT: file %r flagged: %s score=%.4f >= threshold %.2f (engine=%s)%s -> REJECT, pipeline halted",
                extracted.filename,
                file_verdict.label,
                file_verdict.suspicious_score,
                threshold,
                file_verdict.engine,
                file_detail,
            )
            rules = [RULE_JAILBREAK, RULE_FILE_INJECTION]
            result = GuardResult(
                disposition="REJECT",
                flagged=True,
                rules=rules,
                verdict=verdict,
                masking=None,
                masked_prompt=None,
                files=_files_report(files, file_verdicts, None),
            )
            write_flag(
                layer=LAYER_PROMPT_GUARD,
                disposition="REJECT",
                severity="high",
                reason=(
                    "jailbreak or prompt injection detected in attached file "
                    f"{extracted.filename!r}"
                ),
                rules=rules,
                user=user,
                prompt_masked=mask(prompt).masked_text,
                details={
                    "filename": extracted.filename,
                    "label": file_verdict.label,
                    "suspicious_score": file_verdict.suspicious_score,
                    "threshold": threshold,
                    "engine": file_verdict.engine,
                    "matched": file_verdict.matched,
                },
            )
            return result
    logger.info(
        "step 1/2 | RESULT: prompt + %d file(s) below threshold %.2f -> proceed to masking",
        len(files),
        threshold,
    )

    logger.info(
        "step 2/2 | PII masking: scanning prompt + %d file(s) with Presidio",
        len(files),
    )
    combined = assemble_with_files(prompt, files)
    masking_result = mask(combined, reversible=reversible)
    masked_prompt, masked_sections = split_masked(
        masking_result.masked_text, len(files)
    )
    section_entities = {
        index: _section_entities(masking_result.mapping, section)
        for index, section in enumerate(masked_sections)
    }
    files_report = _files_report(files, file_verdicts, section_entities)
    masked_files = [
        (extracted.filename, section)
        for extracted, section in zip(files, masked_sections)
    ]
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
            masked_prompt=masked_prompt,
            files=files_report,
            masked_files=masked_files,
        )
        write_flag(
            layer=LAYER_MASKING,
            disposition="MASKED",
            severity="low",
            reason="personal data detected and masked (prompt and/or attached files)",
            rules=rules,
            user=user,
            prompt_masked=masked_prompt,
            details={
                "entities": masking_result.entities,
                "engine": masking_result.engine,
                "files": files_report,
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
        masked_prompt=masked_prompt,
        files=files_report,
        masked_files=masked_files,
    )
