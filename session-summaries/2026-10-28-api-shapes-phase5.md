# Session Summary — 2026-10-28 · api-shapes · Phase 5: Anthropic Messages conformance

**Plan:** NEXT_SESSION-api-shapes.md (Phase 5)   **Commits:** 35e46c4..6dc8864 (main)
**Deployed:** none (library; pushed to main)   **Model:** Claude Opus 5.5

## Plan vs. actual
Planned: items 1–5 (new grid scenarios, SDK-client cells, count_tokens, typed blocks as input, live check). Shipped: all five, plus a `stream_tools` scenario, `output_config.effort` and `output_config.format` translation, and fixes from an Opus review. Slipped: none.
Scope: expanded twice, both with rdwj's approval: `json_schema` (Phase 5 DoD item 4 requires "the same scenarios as Phase 4", which the session plan's list missed), and `effort` parsing (needed for the thinking scenario to pass strict mode on Chat Completions backends).

## Shipped
- 35e46c4, c0ec61b Messages `output_config.effort` → internal thinking effort (→ `reasoning_effort`); `output_config.format` json_schema → internal response format. Lossless round trips otherwise.
- f1682ae Scenario model carries both wire forms explicitly; `native_form` picks the source of the backend body. Scenarios: system_blocks, tool_error, parallel_tools, image, thinking, stream_usage (replays `stream`). Per-capability models per backend. Shared result checks in `tests/harness/checks.py`.
- b48eee6 `sdk_messages` ingress: `anthropic.AsyncAnthropic` over an httpx2 ASGI client, stream helper + assembled final message checked; `stream_tools` scenario.
- 3dbe58e `POST /v1/messages/count_tokens`, local estimate (`model_calling/token_estimate.py`), documented as one.
- c24e094 `messages()` accepts returned typed blocks; docs example run against Ollama.
- 73dcb91 api-reference: how Messages fields reach non-Anthropic backends.
- a430576, 6dc8864 live Messages tests through Client and gateway (8/8 live pass).
- e190c05 review fixes: `_wire` lost `type` on caller-built blocks; non-list `messages` gave 500/502 instead of 400; offline guard now covers httpx2; httpx2 declared in dev group (pin unchanged).
- 26b6cab json_schema scenario on all rows.

## Verification & confidence
- Offline: 765 passed, 13 strict xfails (was 541/1). Gateway bodies for the 24 pre-existing cells diffed byte-identical before/after (not just pass/fail).
- Live: `pytest -m live` 8/8 on final code. Docs snippets (SDK stream, count_tokens, typed-block tool loop) run against a local gateway/Ollama.
- Fixtures recorded on Ollama, vLLM (gpt-oss-20b, current cluster via port-forward), OpenRouter. Session spend $0.000058; epic total $0.000058 of $5.
- Confidence: high for the grid and translators; medium for count_tokens (estimate constants are arbitrary by design).

## Judgment calls & deviations
- Budget-only thinking stays refused in strict mode on Chat Completions backends; no invented budget→effort mapping (rdwj decided).
- `is_error` stays dropped on text backends, documented (rdwj decided).
- vLLM image cells are strict xfails with no fixture, as the plan asked; review noted they can never XPASS, so they test nothing.
- vLLM gpt-oss returns one tool call per turn: kept the fixture, strict xfail in KNOWN_FAILURES for library/SDK checks; body-match test ignores KNOWN_FAILURES (gateway still equals library there).
- `tool_error` check accepts either explanation or retry; strict replay carries the real claim (history sent correctly).
- Five live calls were first logged at the $0.05 unknown-cost placeholder (OpenRouter generation lookup slower than 10s); corrected to looked-up costs before commit, lookup now waits 30s.
- Sub-agent output for count_tokens and typed blocks was largely rewritten (duck typing, a None crash, hand-built rather than round-trip tests).

## Backlog delta
Filed none · Closed none · Lesson added to CLAUDE.md (anthropic 2.x uses httpx2) · Deferred: Ollama/vLLM ignore effort level (Ollama sends `think: true` only). Missed: Document blocks, listed as still open for Phases 5 and 7, were not done or flagged during the session; carried in the plan.

## Drift & forward-collisions
- Backward: Phase 4 work item 1 (`reasoning_effort`, `response_format` json_schema) partly exercised: the provider side now runs under the grid via Messages; Chat Completions ingress cells for json_schema already pass.
- Forward: Phase 6 Anthropic provider must re-emit `output_config` (effort/format) from the internal request; `_output_config_out` exists for that. Phase 8 (Claude Code) will send budget-only thinking; the gateway is non-strict so it drops, but worth checking.

## For the reviewer
- Sanity-check: the scenario model (explicit per-form fragments) vs. a generated form; and whether `stream_usage` replaying `stream` is too clever.
- Thin verification: count_tokens accuracy (no reference counter); vLLM image coverage is nil.
- Wants guidance: whether vacuous strict xfails for backend capability gaps should become skips.

## Risks / watch-fors
- gpt-oss output varies between recordings; re-recording vLLM can flip cells.
- httpx2 is a fork tied to anthropic's release cadence; SDK upgrades may move it.
