"""Stage 0: file intake for chat uploads - validation, caps, text extraction.

Every attached file is reduced to plain text here before any later guard step
runs: the extracted text is classified by Prompt-Guard (step 1), masked with
Presidio (step 2), screened by the LLM router (step 4), and passed to the
answer LLM as request context. Raw file bytes and extracted text are never
persisted; audit/flag rows record file metadata only.

Deny-by-default: an allowlist of extensions, per-file and total byte caps,
and a per-file character cap (PDFs can decompress to enormous text). Any
intake failure rejects the whole request before screening starts.
"""

import logging
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath

logger = logging.getLogger("guard.file_intake")

ALLOWED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".pdf"}
MAX_FILES = 5
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 15 * 1024 * 1024
MAX_FILE_CHARS = 50_000
TRUNCATION_MARKER = "\n...[truncated]"
#: Per-file excerpt length the LLM router sees (answer LLM sees full text)
ROUTER_EXCERPT_CHARS = 4_000

#: Sentinel between prompt and file sections inside the combined masking
#: text. Indexed digits only (no PII-detectable content), so masking leaves
#: the sentinels intact and the masked combined text can be split back apart.
FILE_SENTINEL = "\n\n<<<GUARD_FILE_{index}>>>\n\n"
_FILE_SENTINEL_SPLIT = re.compile(r"\n\n<<<GUARD_FILE_\d+>>>\n\n")

VALIDATION_SUFFIXES = " ".join(sorted(ALLOWED_EXTENSIONS))


class FileIntakeError(Exception):
    """Aggregated per-file intake errors: ``[{'filename': ..., 'error': ...}]``."""

    def __init__(self, errors: list[dict]):
        self.errors = errors
        super().__init__("; ".join(f"{item['filename']}: {item['error']}" for item in errors))


@dataclass(frozen=True)
class ExtractedFile:
    filename: str  # sanitized display name
    extension: str
    size_bytes: int
    text: str
    error: str | None = None


def sanitize_filename(filename: str) -> str:
    """Reduce a client-supplied name to a safe display name.

    Filenames are client-controlled and end up next to LLM prompts, so strip
    paths, control characters, and anything outside ``[A-Za-z0-9._-]``.
    """
    name = PurePath(str(filename).replace("\\", "/")).name or "file"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "file"
    return name[:120]


def assemble_with_files(prompt: str, files: list[ExtractedFile]) -> str:
    """Join the prompt and every file text with indexed sentinels."""
    parts = [prompt]
    for index, extracted in enumerate(files, start=1):
        parts.append(FILE_SENTINEL.format(index=index) + extracted.text)
    return "".join(parts)


def split_masked(masked_combined: str, file_count: int) -> tuple[str, list[str]]:
    """Split a masked combined text back into ``(prompt, [file sections])``.

    Degrades to returning the whole text as the prompt part (and empty file
    sections) if the sentinel count ever mismatches, instead of losing text.
    """
    parts = _FILE_SENTINEL_SPLIT.split(masked_combined)
    if len(parts) != file_count + 1:
        logger.error(
            "file_intake | sentinel split mismatch: %d parts for %d file(s)",
            len(parts),
            file_count,
        )
        return masked_combined, [""] * file_count
    return parts[0], parts[1:]


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_text(extension: str, data: bytes) -> str:
    if extension == ".pdf":
        return _extract_pdf(data)
    return data.decode("utf-8")


def intake_file(filename: str, data: bytes) -> ExtractedFile:
    """Validate and extract one uploaded file; failures land in ``error``."""
    safe_name = sanitize_filename(filename)
    extension = PurePath(safe_name).suffix.lower()
    size = len(data)
    if extension not in ALLOWED_EXTENSIONS:
        return ExtractedFile(
            safe_name,
            extension,
            size,
            "",
            f"unsupported file type; allowed: {VALIDATION_SUFFIXES}",
        )
    if size > MAX_FILE_BYTES:
        return ExtractedFile(
            safe_name,
            extension,
            size,
            "",
            f"file exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB limit",
        )
    if size == 0:
        return ExtractedFile(safe_name, extension, size, "", "empty file")
    try:
        text = _extract_text(extension, data).strip()
    except UnicodeDecodeError:
        return ExtractedFile(safe_name, extension, size, "", "not valid UTF-8 text")
    except Exception as exc:
        return ExtractedFile(safe_name, extension, size, "", f"extraction failed: {exc}")
    if not text:
        return ExtractedFile(safe_name, extension, size, "", "no extractable text")
    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + TRUNCATION_MARKER
    return ExtractedFile(safe_name, extension, size, text)


def intake_files(uploads: list[tuple[str, bytes]]) -> list[ExtractedFile]:
    """Intake every upload; raise :class:`FileIntakeError` listing all bad files."""
    if len(uploads) > MAX_FILES:
        raise FileIntakeError(
            [
                {
                    "filename": "(request)",
                    "error": f"too many files: {len(uploads)} > {MAX_FILES}",
                }
            ]
        )
    if sum(len(data) for _, data in uploads) > MAX_TOTAL_BYTES:
        raise FileIntakeError(
            [
                {
                    "filename": "(request)",
                    "error": (
                        "total upload size exceeds the "
                        f"{MAX_TOTAL_BYTES // (1024 * 1024)} MB limit"
                    ),
                }
            ]
        )
    extracted = [intake_file(name, data) for name, data in uploads]
    errors = [
        {"filename": item.filename, "error": item.error}
        for item in extracted
        if item.error
    ]
    if errors:
        raise FileIntakeError(errors)
    for item in extracted:
        logger.info(
            "file_intake | %s: %d bytes -> %d chars",
            item.filename,
            item.size_bytes,
            len(item.text),
        )
    return extracted
