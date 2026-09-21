# Remove files & dead code not needed by the active APIs

## Goal

Slim the project to what the **active (uncommented) endpoints** need: `/v1/health`, `/v1/token`, `/v1/users/me`, `/v1/users`, `/v1/chat`, `/v1/chat/upload`, `/v1/audit` (plus the lifespan warmup, which calls `screen("warmup")` and `embed("warmup")` — keep both).

## Confirmed decisions

1. **Keep** `guard/ingest.py`, `tests/test_ingest.py`, `data/medicines.json`, `data/patients.json` — not imported by the API, but the only way to populate the `documents`/`chunks` tables that `/v1/chat` retrieval queries.
2. **Strip in-file dead code** tied to the commented-out `/v1/screen`, `/v1/ask`, `/v1/documents` endpoints (user chose full strip, not files-only).

## Import closure of active endpoints (keep, no changes except noted)

`guard/`: `__init__.py`, `api.py`, `auth.py`, `abac.py`, `chat.py`, `db.py`, `flags.py`, `ingest.py`, `llm.py`, `logconf.py`, `pipeline.py`, `steps/` (all files). Tests: everything except the `test_rag.py` rework below.

## Steps

### 1. Delete files

- `guard/cli.py` — terminal screening UI; only imported by `guard/__main__.py`
- `guard/__main__.py` — CLI entry point (`python -m guard`); nothing else references it
- `flags.jsonl.bak` — stale backup, referenced by nothing
- Optionally delete `guard/__pycache__/` and `tests/__pycache__/` (untracked caches; contain stale `.pyc` for the old flat layout `rag`/`prompt_guard`/`masking` and for deleted `__main__`)

Use `git rm` for the three tracked files so the deletion is staged.

### 2. `guard/api.py` — strip dead code

- Imports:
  - Delete `from guard.steps.rag import AskResult, ask` (line 32)
  - Delete `func` from the `sqlalchemy` import (line 20; only the commented `/documents` block used it)
  - Delete `Chunk` and `Document` from the `guard.db` import (line 30; only used by commented code — keep `AuditLog`, `User`, `get_session`, `init_db`)
- Delete models used only by commented endpoints: `ScreenRequest`, `ScreenResponse`, `AskRequest`, `RetrievedChunkOut`, `AskResponse`, `DocumentOut`
- Delete the `CLEAN_MESSAGE` constant (only the commented `/screen` block used it; keep `REJECT_MESSAGE`, `LLM_REJECT_MESSAGE`, `ABAC_REJECT_MESSAGE`, `MASKED_MESSAGE` — all used by `_chat_response`)
- Delete the three commented endpoint blocks (`# @public_router.post("/screen"...)`, `# @public_router.post("/ask"...)`, `# @public_router.get("/documents"...)`) and the `_ask_response()` helper (lines ~374–402; only the commented `/ask` called it)
- Keep: `CitationOut` (used by `ChatResponse`), `GuardResult`/`screen` import (lifespan warmup), `embed` import (warmup)

### 3. `guard/steps/rag.py` — remove `ask()`

- Delete the `AskResult` dataclass (lines ~89–98) and `ask()` (lines ~291–394)
- Remove imports/members that become unused **inside rag.py**:
  - `from guard.pipeline import screen` (only `ask` called it; `retrieve` does not — also drops this module's edge in the rag↔pipeline cycle)
  - `POLICY_VERSION` from `guard.abac`, `write_flag`, `LAYER_ABAC`, `EVENT_RAG_QUERY` (import and the `EVENT_RAG_QUERY = EVENT_RAG_QUERY` re-export on line 43)
  - Constants `RULE_ABAC_ATTEMPT`, `REJECT_MESSAGE`, `ABAC_REJECT_MESSAGE`, `EMPTY_MESSAGE`
- Keep: `retrieve()`, `RetrievalOutcome`, `RetrievedChunk`, `Citation`, `RULE_RAG_INJECTION`, `append_flag`, `user_block`, `LAYER_PROMPT_GUARD`, `EVENT_RAG_INJECTION`, `classify`, `_scan_chunk`, `_restricted_attempt`, `_permitted_rows`, `_cosine`, `get_top_k`, `get_abac_threshold`, thresholds
- Update the module docstring (currently describes the `ask()` flow) to describe `retrieve()` only
- Leave `guard/flags.py` untouched (`EVENT_RAG_QUERY` stays defined there as part of the flag schema; nothing else imports it)

### 4. `tests/test_rag.py` — convert from `ask()` to `retrieve()`

Every current test drives `ask()`; `retrieve()` is still live code used by chat and must keep direct coverage (test_chat.py mostly monkeypatches `retrieve`).

- Change `from guard.steps.rag import ask` to `from guard.steps.rag import retrieve`; call `rag.retrieve(session, user, query)` directly (queries are "already masked by the caller" now)
- Convert and keep: ABAC differential (user1 public-only, admin gets patient chunks), engine mismatch (set chunk engines to `minilm`), empty index, poisoned-chunk drop + `RAG_CONTEXT_INJECTION` flag row, `GUARD_RAG_TOP_K` default, unauthorized-attempt restricted scoring (`unauthorized_attempt`, `restricted_top_score`, `restricted_match_ids` on `RetrievalOutcome`), admin-query-not-an-attempt
- Drop ask-only tests: screen-REJECT halts before retrieval, masked-prompt-gets-embedded, `RAG_QUERY` audit-row test (chat writes audit rows itself now)
- In the `offline_engines` fixture, drop the `rag.screen` monkeypatch; keep the `classify`/embedding stubs

### 5. `README.md` — update references

- Project layout tree: remove `cli.py` and `__main__.py` lines; change `rag.py` comment to "retrieve() core: ABAC-filtered retrieval (used by /v1/chat)"; update `test_rag.py` description
- Remove the `python -m guard` terminal-usage sections/examples (lines ~113, ~119, ~490, ~510, ~529)
- Remove/adjust `/v1/ask`, `/v1/screen`, `/v1/documents` sections and examples (~7, ~103, ~162–163, ~188, ~204, ~246, ~260, ~306, ~572): the "currently disabled" note now means the code is gone; Stage 3 doc should describe retrieval as part of `/v1/chat` only
- Keep docs for `/v1/chat`, `/v1/chat/upload`, `/v1/audit`, `/v1/token`, `/v1/users`, ingest, chat pipeline, Gemini client

## Validation

From `custom/` with the project venv:

1. `.venv\Scripts\python -m pytest tests -q` — full suite green (no collection errors, no skipped-as-missing imports)
2. `.venv\Scripts\python -c "from guard.api import app; print([r.path for r in app.routes])"` — app imports cleanly, only the 7 active routes + docs
3. Grep guards (must return nothing in `guard/` and `tests/`): `guard.cli`, `__main__`, `_ask_response`, `AskResult`, `AskResponse`, `ScreenRequest`, `ScreenResponse`, `DocumentOut`, `CLEAN_MESSAGE`, `import ask`, `from guard.steps.rag import ask`
4. `git status` — only the intended deletions/edits

## Risks / notes

- The `test_rag.py` rewrite is the largest change; reuse its `_seed`/`_user`/fixtures as-is
- Do not touch `guard/ingest.py`, `data/`, or `tests/test_ingest.py`
- The commented endpoint blocks are deleted outright per the user's decision; re-enabling them later means restoring code from git history
- `.venv/`, `.env`, `.env.example`, `requirements.txt`, `.gitignore`, `.kilo/` stay untouched
