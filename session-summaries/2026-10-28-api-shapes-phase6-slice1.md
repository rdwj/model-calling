# Session Summary — 2026-10-28 · api-shapes · Phase 6 slice 1: the Anthropic Messages provider

**Plan:** NEXT_SESSION-api-shapes.md   **Commits:** c777d03..e155148 (main, unpushed at close)
**Deployed:** none   **Model:** Claude Opus 5.5 (sub-agents: Sonnet for provider unit tests, Opus for review)

## Plan vs. actual
Planned: Anthropic provider, `vllm_messages` as an in-grid column with a lossless-pass-through assertion, Messages tool `strict` to Chat Completions backends, a live OpenRouter Anthropic column. Shipped: all four, plus fixes for three review bugs. Slipped: none.
Scope: expanded in two places, each to make item 2 possible: Chat Completions ingress now sets `content_shorthand`, and the harness gained per-ingress-form fixtures for Messages-speaking backends. Item 3 was done as a typed `Tool.strict` rather than the planned extras copy.

## Shipped
- c777d03: Chat Completions string content is marked shorthand; `block_to_internal` and `stop_reason_to_internal` in `formats.anthropic`.
- 37721fe: `AnthropicProvider` and its SSE parser for `anthropic/`. Legacy adapter and Claude 3 tables removed. `vllm_messages` is in the grid (19 fixtures, recorded free). Per-form fixtures. Pass-through test.
- af775c3: `Tool.strict` typed; tool_strict xfails removed (51 → 35 xfails).
- 1de8f73: live Anthropic column on OpenRouter `/api/v1/messages` (8 calls per run); cost taken from `usage.cost`.
- e155148: review fixes: an assistant turn left empty by the unsigned-thinking drop is now removed; an unknown effort no longer turns into adaptive thinking; stream usage merges cache counts from `message_start`. Also no `tool_choice` without tools, and a harness check for leftover per-form fixtures.

## Verification & confidence
- Offline: 1323 passed, 35 xfailed with an empty HOME. Each commit was checked in its own worktree (all green, black clean). CI lint (flake8 subset) is clean. CI hasn't run on these commits: they're unpushed.
- Live: `pytest -m live` passed 22/22. The Anthropic provider drove OpenRouter's Messages endpoint for text, tools, a streamed tool call, and thinking, through both the library and the gateway (SDK).
- Confidence: high for vLLM and OpenRouter. Untested against Anthropic's own API: no key exists, and the Bearer auth choice rests on the SDK's behavior.

## Judgment calls & deviations
- Auth is `Authorization: Bearer`, not the plan's `x-api-key`: anthropic 2.0.0 sends keys that way (`_client.py` `_api_key_auth`). `anthropic-version` is the SDK's 2026-10-21.
- vLLM ends a forced tool call with `end_turn`; the provider reports `tool_use` (same as the OpenAI-compatible provider). A response repair, not a pass-through.
- Chat Completions `reasoning_effort` maps to adaptive thinking plus `output_config.effort`; missing `max_tokens` becomes 4096.
- The live thinking cell uses claude-haiku-4.5: Haiku 5.5 on OpenRouter returned no thinking for any config.
- Unsigned thinking is dropped. Thinking vLLM signed can't be told apart and is still sent (Anthropic would reject it). Documented.

## Backlog delta
Filed none · Closed none · Lessons added to CLAUDE.md (OpenRouter `msg_` cost lookup, Haiku 5.5 no thinking) · Deferred from review (low): Chat Completions extras (`logprobs`, `user`, `logit_bias`) silently dropped by the Anthropic provider instead of refused; streamed `stop_details`/`container` lost; Chat-form `reasoning_effort` against older Claude models unverified live; no cell pins `max_completion_tokens` → Messages `max_tokens` (vllm_messages refuses that scenario for `seed`); `/v1/models` lists no `anthropic/` models.

## Drift & forward-collisions
- Backward: Phase 10 (unauthenticated `/v1/config/update`) now also guards `ANTHROPIC_API_KEY` through the new provider: still valid, slightly more urgent. The "Document block types" open item is unchanged.
- Forward: Phase 9 pass-through, for vLLM/OpenRouter Messages, is largely satisfied for the Messages ingress, since the provider sends the client's body unchanged (the request still goes through the internal format). Phase 8 (Claude Code acceptance, backend a) can use `anthropic/` with base_url set to OpenRouter as in `tests/live/test_anthropic_live.py`. No issue tracker items to comment on; noted for /plan-next-session.

## For the reviewer
- Sanity-check: the per-form fixture mechanism (`Backend.fixture_name`, `record._fixture_names`) and the decision to mark Chat Completions string content as shorthand at ingress.
- Thin verification: Anthropic's own API (auth, `2026-10-21` version, adaptive thinking from `reasoning_effort`) is untested.
- Wants guidance: whether the provider should refuse Chat Completions-only extras (`logprobs`, `logit_bias`) in strict mode, as it already does for seed and the penalties.

## Risks / watch-fors
- The ledger shows $0.0527 epic spend, but $0.05 is one unknown-cost placeholder from a slow mistral-nemo lookup; real spend is about $0.003. Placeholders inflate the budget guard over time.
- The vLLM port-forward dies between sessions; restart it from rdwj's `~/.zshrc` comment.
- A `git stash -u` round trip during close re-wrote the untracked `test_model_calling.py` (content unchanged, mtime now later).
