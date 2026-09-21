"""Offline tests for stage 0 file intake: validation, caps, extraction."""

import pytest

from guard.steps.file_intake import (
    MAX_FILES,
    MAX_FILE_BYTES,
    MAX_FILE_CHARS,
    TRUNCATION_MARKER,
    ExtractedFile,
    FileIntakeError,
    assemble_with_files,
    intake_file,
    intake_files,
    sanitize_filename,
    split_masked,
)


def _pdf_with_text(text: str) -> bytes:
    """Minimal valid single-page PDF rendering `text` via a Tj operator."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF"
    ).encode()
    return bytes(out)


def test_sanitize_filename_strips_paths_and_unsafe_chars():
    assert sanitize_filename(r"..\..\etc\secret report.txt") == "secret_report.txt"
    assert sanitize_filename("/tmp/passwd") == "passwd"
    assert sanitize_filename("weird <name>?.txt") == "weird_name_.txt"
    assert sanitize_filename("") == "file"
    assert sanitize_filename("...") == "file"
    assert len(sanitize_filename("a" * 500 + ".txt")) == 120


def test_intake_file_plain_text():
    result = intake_file("notes.txt", b"hello world")
    assert result.error is None
    assert result.filename == "notes.txt"
    assert result.extension == ".txt"
    assert result.size_bytes == 11
    assert result.text == "hello world"


def test_intake_file_unsupported_extension():
    result = intake_file("payload.exe", b"MZ...")
    assert result.error is not None
    assert "unsupported file type" in result.error
    assert ".txt" in result.error and ".pdf" in result.error


def test_intake_file_oversize():
    result = intake_file("big.txt", b"a" * (MAX_FILE_BYTES + 1))
    assert result.error is not None
    assert "MB limit" in result.error


def test_intake_file_empty():
    result = intake_file("empty.txt", b"")
    assert result.error == "empty file"


def test_intake_file_invalid_utf8():
    result = intake_file("binary.txt", b"\xff\xfe\x00\x01")
    assert result.error == "not valid UTF-8 text"


def test_intake_file_corrupt_pdf():
    result = intake_file("broken.pdf", b"%PDF-1.4 this is not really a pdf")
    assert result.error is not None
    assert "extraction failed" in result.error


def test_intake_file_pdf_extraction():
    result = intake_file("report.pdf", _pdf_with_text("Hello file world"))
    assert result.error is None
    assert result.extension == ".pdf"
    assert "Hello file world" in result.text


def test_intake_file_whitespace_only_is_error():
    result = intake_file("blank.md", b"   \n\t  ")
    assert result.error == "no extractable text"


def test_intake_file_truncates_to_char_cap():
    result = intake_file("long.txt", b"x" * (MAX_FILE_CHARS + 1000))
    assert result.error is None
    assert len(result.text) == MAX_FILE_CHARS + len(TRUNCATION_MARKER)
    assert result.text.endswith(TRUNCATION_MARKER)


def test_intake_files_happy_path():
    results = intake_files(
        [("a.txt", b"one"), ("b.json", b'{"k": 1}'), ("c.pdf", _pdf_with_text("pdf body"))]
    )
    assert [r.filename for r in results] == ["a.txt", "b.json", "c.pdf"]
    assert "pdf body" in results[2].text


def test_intake_files_aggregates_every_error():
    with pytest.raises(FileIntakeError) as excinfo:
        intake_files([("good.txt", b"fine"), ("bad.exe", b"MZ"), ("broken.pdf", b"junk")])
    names = [item["filename"] for item in excinfo.value.errors]
    assert names == ["bad.exe", "broken.pdf"]
    assert all(item["error"] for item in excinfo.value.errors)


def test_intake_files_rejects_too_many_files():
    uploads = [(f"f{i}.txt", b"x") for i in range(MAX_FILES + 1)]
    with pytest.raises(FileIntakeError) as excinfo:
        intake_files(uploads)
    assert excinfo.value.errors[0]["filename"] == "(request)"
    assert "too many files" in excinfo.value.errors[0]["error"]


def test_intake_files_filename_sanitized():
    results = intake_files([(r"..\drop table.txt", b"clean")])
    assert results[0].filename == "drop_table.txt"


def test_assemble_and_split_round_trip():
    files = [
        ExtractedFile("a.txt", ".txt", 3, "alpha"),
        ExtractedFile("b.txt", ".txt", 4, "beta"),
    ]
    combined = assemble_with_files("the prompt", files)
    assert "the prompt" in combined and "alpha" in combined and "beta" in combined
    prompt, sections = split_masked(combined, 2)
    assert prompt == "the prompt"
    assert sections == ["alpha", "beta"]


def test_split_masked_with_placeholders_inside_sections():
    masked = "p [REDACTED_1]\n\n<<<GUARD_FILE_1>>>\n\nf [REDACTED_2] tail\n\n<<<GUARD_FILE_2>>>\n\n[REDACTED_1]"
    prompt, sections = split_masked(masked, 2)
    assert prompt == "p [REDACTED_1]"
    assert sections == ["f [REDACTED_2] tail", "[REDACTED_1]"]


def test_split_masked_mismatch_degrades_to_whole_prompt():
    prompt, sections = split_masked("no sentinels here", 2)
    assert prompt == "no sentinels here"
    assert sections == ["", ""]
