# model-calling

LLM gateway and client library: OpenAI Chat Completions, OpenAI Responses,
and Anthropic Messages in and out, over Ollama, vLLM, OpenAI-compatible
(including OpenRouter), Anthropic, and Cohere backends.

Active epic plans live in `NEXT_SESSION-*.md` at the repo root and are
tracked in git.

## Rules

- **Clean room.** The library API is meant to feel familiar to LiteLLM
  users, but never read, copy, or adapt LiteLLM source, docs, types, or data
  files. Work from provider specs and the official `openai` / `anthropic`
  SDK types. See "Independent Implementation" in `CONTRIBUTING.md`.
- **Check current SDK types before writing translators.** Both SDKs move
  quickly (`openai` 3.x and `anthropic` 2.x as of 2026-10). Look them up;
  don't work from memory.
- **Offline by default.** Plain `pytest` needs no network, backend, or files
  outside the repo. Live tests (`-m live`) use `OPENROUTER_API_KEY` from the
  environment and append their cost to the committed
  `tests/live/spend.jsonl`. Respect the budget in the active epic plan.
- **Dependency policy.** Every external dependency is pinned to an exact
  version (`==`, commit SHA, or image digest) that is at least 7 days old and
  has no known major CVEs, enforced by `[tool.uv] exclude-newer` and
  `pip-audit` in CI. Never add LiteLLM. See "Dependency Policy" in
  `CONTRIBUTING.md`.
- **No local-only dependencies.** Anything the plans or tests rely on must
  be committed, fetched from a public source, or read from an environment
  variable.
- Scrub auth headers from recorded fixtures and run gitleaks before
  committing them. This repo is public.
- **Grid xfails are the bug list.** Offline coverage goes in the
  (ingress, backend, scenario) grid (`tests/harness/grid.py`). Known-wrong
  behavior is a strict xfail in `KNOWN_FAILURES` naming the fixing phase;
  remove the entry when the fix lands. OpenRouter diverges from the SDK
  types in small ways; record each one in `tests/fixtures/README.md`
  instead of loosening validators.
