# Session Summary — 2026-10-28 · api-shapes · Phase 2 slice 2: OpenAI-compatible providers, legacy stream path removed

**Plan:** NEXT_SESSION-api-shapes.md (Phase 2 slice 2)   **Commits:** fd2679c..f986e98 (main)
**Deployed:** none (library and gateway; pushed to origin through 21c44fb)   **Model:** Claude Opus 5.5

## Plan vs. actual
Planned: record vLLM fixtures, port vLLM/OpenAI/OpenRouter onto one provider, delete `server.stream_response`, fix the remaining Phase 2 bugs. Shipped: all four items plus a round of fixes from an independent review. Slipped: none.
Scope: stayed in Chat Completions ingress. Small expansions: the OpenAI embedding adapter's URL (forced by the `base_url` change), a `headers` config key (to restore `OPENAI_ORG_ID`), and a doc pass on the vLLM config examples.

## Shipped
- `fd2679c` vLLM fixtures from gpt-oss-20b (vLLM 0.24). The recorder reads `VLLM_*` from the environment and stores placeholder URL/model, adds an opt-in `VLLM_VERIFY_TLS=false`, and builds native bodies through the provider.
- `0317495` `OpenAICompatProvider` (+ OpenAI/vLLM/OpenRouter subclasses), SSE parser, pooled `HttpClients` with injectable transport (no more `httpx.AsyncClient.__init__` patching), backend 401/403 → 502, legacy `VLLMAdapter` and `/generate` mode deleted. `KNOWN_FAILURES` emptied.
- `3202417` `stream_response` and the keyword reasoning heuristic deleted; Anthropic/Cohere streaming → 501 naming Phase 6.
- `e88f937` / `f0ad82f` / `23e024a` item 4: no `basicConfig` on import, unique legacy response ids, moderation kept off the chat registry.
- `21c44fb` review fixes: system blocks joined with newlines, streamed refusals, `OpenAI-Organization` via `headers`, reused tool-call index 0, `top_k` via extras, finish without `[DONE]`, null model/message, uncapped pool.
- `f986e98` test README.

## Verification & confidence
- Offline: `HOME=$(mktemp -d) uv run pytest` → 275 passed. Grid cells for ollama, openai_compat, and vllm all pass against recorded fixtures. Every item 4 bug and every review fix has a test that was run against the unfixed code and failed.
- Live: the in-process gateway against the real vLLM endpoint (scratch script, not committed): text, tools, stream, and missing-model all correct and SDK-valid.
- CI on 21c44fb was in progress at close. Local suite, black, and flake8 were clean.
- Confidence: high for vLLM and OpenRouter shapes (real recordings), medium for `OpenAIProvider`-specific behavior (unit tests only, no OpenAI recording or live call).

## Judgment calls & deviations
- `base_url` for vllm/openai/openrouter is now the API root with `/v1` (openai SDK convention). This breaks vLLM configs; recorded for the Phase 9 migration guide.
- OpenAI gets `max_completion_tokens`, no `top_k` (raises), and drops reasoning on assistant turns. vLLM/OpenRouter send `reasoning` and `max_tokens`. These are assumptions from the specs, not tested against OpenAI.
- `openai_compat` grid backend now routes as `openrouter/…` so it exercises the named provider's default URL.
- Scenario `max_tokens` raised 64 → 256: gpt-oss used 64–67 tokens on reasoning alone. The older Ollama and OpenRouter fixtures still record 64; replay ignores bodies.
- Only `create_openai_response` got uuid ids; the streaming helper and `OllamaAdapter` builders keep `int(time.time())` because nothing in the server reaches them now.
- Moderation always uses Granite Guardian; the `model` field no longer selects a backend.
- `/v1/models/refresh` for `vllm/…` now 404s (no legacy adapter).

## Backlog delta
Filed none · Closed none · Phase 9 migration-guide list extended in the plan (vLLM `/generate`, `base_url`, 401/403 → 502) · Deferred: request-body matching in replay (see Risks).

## Drift & forward-collisions
- Backward: none (no open issues).
- Forward: Phase 3 "pooled httpx client" on `Client` partly exists as `providers.HttpClients` (owned by the registry). Phase 3 should build on it rather than add a second pool. Phase 3 "exceptions carry provider, status" can start from `BackendError`.

## For the reviewer
- Sanity-check: the `base_url` convention change and the 401/403 → 502 mapping, both user-visible breaks.
- Thin verification: vLLM was exercised only through its OpenAI Chat Completions endpoint; its native Anthropic `/v1/messages` was probed by hand after close (text, tools, and streaming all work) and is queued for recording in Phase 3 slice 1. `OpenAIProvider` (no fixture or live call); pool limit change (untested by design); `/v1/models/refresh` regression for vllm.
- Wants guidance: whether the grid should compare request bodies against recorded ones before Phase 3/4 add more cells.

## Risks / watch-fors
- An empty `KNOWN_FAILURES` proves less than it looks. Replay matches on URL only, so the body the provider sends is never checked. No grid cell uses `OpenAIProvider`. `assert_backend_auth` covers only `openai_compat`.
- gpt-oss output varies between recordings (the first stream recording at 64 tokens had no answer). Re-recording vLLM can flip cells; check the fixture before committing.
- Sub-agent output needed review again: one sub-agent's test asserted a double-slash URL as correct and hardcoded recording-specific ids. Its claim that the files were "well under 512 lines" was also false.
