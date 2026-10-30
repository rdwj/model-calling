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

## Lessons

- **anthropic 2.x runs on `httpx2`**, its fork of httpx. `AsyncAnthropic(http_client=...)`
  rejects an `httpx.AsyncClient`; give it `httpx2.AsyncClient(transport=httpx2.ASGITransport(app=...))`
  to drive the in-process gateway, as `tests/test_sdk_grid.py` does.
- **Ollama's `think` values are per model.** `/api/show` returns `thinking.values`: gpt-oss
  takes only `"low"|"medium"|"high"`, qwen3:4b only `true`. An unknown level silently falls back
  to the model default (checked on Ollama 0.40.12), so never guess; the provider looks it up.
- **The gateway drops unsupported params by default**; only the library is strict. Grid cells
  that assert a strict refusal through the gateway set `server.DROP_PARAMS = False`.
- **OpenRouter's generation cost lookup can lag over 10s.** The ledger polls up to 30s;
  a `0.05` entry in `spend.jsonl` is the unknown-cost placeholder, not a real charge.
- **OpenRouter's Messages endpoint returns `msg_` ids, which its generation lookup rejects.**
  Take the cost from `usage.cost` (on the body, or the stream's `message_delta`) instead, as
  `tests/live/test_anthropic_live.py` does through a callback.
- **claude-haiku-5.5 on OpenRouter never thinks** (`thinking_tokens: 0` for enabled or adaptive,
  checked 2026-10-28); use claude-haiku-4.5 for live thinking.
- **Claude Code `--bare` offers only Bash, Edit, and Read** (2.1.308; `--tools Write` doesn't add
  it). Edit creates a file when `old_string` is empty, but models don't assume so; say it in the
  task. `--allowedTools` is variadic and swallows a trailing prompt, so pass the prompt on stdin.
- **Claude Code always asks for thinking** (`adaptive`, `display: omitted`) plus
  `context_management` and `output_config.effort`, and puts a device id in `metadata.user_id`.
  Never commit a captured Claude Code request body.
- **Mount the alias file's directory, never the file.** A single-file bind mount (compose) or a
  `subPath` mount (Kubernetes) pins the original inode, so an editor's rename-on-save or a ConfigMap
  update never reaches the container and hot reload silently stops working.
- **Tell workflow reviewers the expected `Assisted-by` trailer.** A Session D reviewer read the
  global CLAUDE.md's `Sonnet 4.5` example as a requirement, flagged the correct Opus trailer as
  must-fix, and burned the stream's only fix pass on it. State the trailer in the reviewer brief.
