# Next Session — api-shapes

## Next: Phase 3, slice 1: the library API in both formats, with streaming

Make model-calling importable in place of LiteLLM: a `Client` with
`completion`/`acompletion` (OpenAI types) and `messages`/`amessages`
(Anthropic types), streaming and non-streaming, over the providers Phase 2
built. Tighten the grid first so the new cells prove what they claim.
rdwj confirmed all four Phase 3 design proposals on 2026-10-28 and chose to
include streaming in this slice, which pulls the Anthropic SSE encoder
forward from Phase 5.

0. **Grid blind spots** (mechanical, do first). Replay matches on URL only,
   so the body a provider sends is never checked. Make `ReplayTransport`
   compare the sent JSON to the fixture's recorded request, with an
   explicit, short ignore list for fields that legitimately vary. Older
   Ollama and OpenRouter fixtures were recorded with `max_tokens: 64` and
   no `stream_options`. Re-record Ollama (free, needs `ollama pull
   qwen3:4b`) rather than widening the ignore list. Ask before
   re-recording OpenRouter (paid); if declined, ignore those two fields
   for `openai_compat` only and say so in `tests/fixtures/README.md`. Run
   `assert_backend_auth` for every backend that has a key.

0b. **Record vLLM's Anthropic Messages endpoint** (free, needs the
   port-forward). vLLM serves `POST /v1/messages` natively; probed by hand
   2026-10-28 on gpt-oss-20b: text (a `thinking` block with a hex
   `signature`, then `pong`, `end_turn`), tools (`tool_use` for
   get_weather/Paris), and streaming (the full `message_start` …
   `message_stop` event sequence). Teach the recorder a Messages-shaped
   native body and add a `vllm_messages` backend with the four scenarios,
   same placeholder scrubbing as `vllm`. Validate the bodies and events
   against the `anthropic` SDK types and record divergences in
   `tests/fixtures/README.md` (for example `chatcmpl-` ids and
   vLLM-minted signatures). These fixtures can't join the grid until an
   Anthropic provider exists (Phase 6); this session they are the
   reference for item 3's encoder.

1. **Errors and `Client`.** An exception hierarchy with OpenAI-style names
   (`AuthenticationError`, `PermissionDeniedError`, `NotFoundError`,
   `RateLimitError`, `BadRequestError`, `APIConnectionError`,
   `APITimeoutError`, `InternalServerError`), each carrying provider,
   status, and raw body. `BackendError` maps onto it. The library raises
   the backend's real status; the gateway's 401/403 → 502 stays a server
   concern. `Client(...)` owns per-prefix config, callbacks, and an
   `HttpClients` pool (reuse `providers.HttpClients`, don't add a second
   one), with `async with` / `aclose`. It routes with `provider_for` and
   takes per-call `api_base` / `api_key`. Module-level functions use a
   lazily created default client.

2. **`completion` / `acompletion`.** Chat Completions in and out, reusing
   `formats/chat_completions.py` and `ChatCompletionsStreamEncoder`.
   Typed pydantic responses of our own that validate against
   `openai.types.chat.ChatCompletion` / `ChatCompletionChunk` in tests.
   The `openai` and `anthropic` SDKs stay dev-only. Streaming returns an
   async iterator of chunks; the sync form returns a sync iterator.

3. **`messages` / `amessages` and the Anthropic SSE encoder.** Messages in
   via `formats/anthropic.py`, out as Anthropic types. A new encoder turns
   internal events into the full event sequence (`message_start`,
   `content_block_start`, `content_block_delta` with `text_delta` /
   `input_json_delta` / `thinking_delta`, `content_block_stop`,
   `message_delta`, `message_stop`), validated against the `anthropic`
   SDK's stream event types. The open item from slice 1, thinking with no
   signature on Anthropic egress, surfaces here; decide it with rdwj if it
   blocks a cell rather than inventing a signature.

4. **Strict by default.** Where a provider can't honor a parameter today
   it warns or drops it. Ollama's forced `tool_choice` and
   `cache_control`, and the OpenAI-compatible providers' token-budget
   thinking, are known cases. Make each a `BadRequestError` naming the
   parameter, with `drop_params=True` on the client or call as the opt-in.
   The gateway keeps its current behavior until slice 2 decides how the
   server exposes it.

**Sequencing.** 0 and 0b, then 1 (2 and 3 need it), then 2, then 3. 4 last: it is
small once the error types exist. If the session runs long, stop after 3
with 4 written up; don't cut streaming from 2 or 3.

**Constraints for the session:**
- Exit predicate: `HOME=$(mktemp -d) uv run pytest` is green. The grid has
  `library_completion` and `library_messages` ingresses over every
  backend and scenario, with no `KNOWN_FAILURES` entries for them. Replay
  rejects a body mismatch, shown by a test that fails on a tampered
  fixture. `git grep -nE "^(import|from) (openai|anthropic)" model_calling`
  returns nothing.
- Scope: library only. The `[server]` extra, server-over-core refactor,
  and `POST /v1/messages` are slice 2. Don't change gateway behavior
  except where item 0's stricter replay exposes a real bug.
- No new runtime dependencies (httpx and pydantic only for this code).

**Session start protocol:**
- Premise checks (before item 0, report before acting): `main` matches
  `origin/main` at the plan commit after `2847c81`. "Test Package" is green
  on `21c44fb` or later (it was still running at planning time).
  `HOME=$(mktemp -d) uv run pytest` shows 275 passed. `ollama list`
  shows `qwen3:4b` if item 0 re-records Ollama.
- Rules with history:
  - Never loosen a grid check or widen replay's ignore list to make a cell
    pass; strict xfails decide.
  - Read sub-agent test code before committing. Last session a sub-agent
    asserted a double-slash URL as correct, hardcoded recording-specific
    ids, and misreported its file sizes.
  - Check installed SDK types (openai 3.35.1, anthropic 2.0.0) before
    writing the response models and the Anthropic encoder, not memory.
- Stop-and-ask before: any OpenRouter spend (re-recording `openai_compat`
  included); if the vLLM port-forward or model is down, ask rdwj rather
  than skipping 0b, changing `[project] dependencies`, moving the dependency
  cutoff, force-pushing, or any change to the vLLM deployment.
- Close ritual: `/session-close`, then `/plan-next-session` (slice 2:
  `[server]` extra, server as a thin layer over `Client`, `POST
  /v1/messages`, docs).

## Remaining epic phases

End-state: model-calling replaces LiteLLM in rdwj's projects and treats the
OpenAI and Anthropic wire formats as equals. Apps are about as likely to
speak one as the other. The library API offers `completion()` (OpenAI types)
and `messages()` (Anthropic types) side by side. It is familiar to LiteLLM
users but not bound to LiteLLM's design; breaking changes are acceptable
where they are an improvement, and the migration guide documents them. The
optional server speaks OpenAI Chat Completions, Anthropic Messages, and
OpenAI Responses (stateless). Every ingress works against every backend:
Ollama, vLLM, OpenAI-compatible (including OpenRouter), Anthropic, and
Cohere v2. An offline test suite enumerates that ingress × backend grid and
validates output against the official `openai` and `anthropic` SDK types.
Live smoke tests run through OpenRouter, and Claude Code completes a
tool-using task through the gateway. The epic ends with 0.2.0 on PyPI and a
pilot project migrated.

### Phase 1: Offline test harness

Make the code testable without a running server or backend, so every later
phase has a deterministic exit predicate.

**Work:**
1. Add the `openai` and `anthropic` SDKs to the `dev` dependency group under
   the dependency policy (exact pins, at least 7 days old, clean audit).
   Look up their current type modules and write validator helpers that parse
   our output with them.
2. Use httpx transport replay for backend responses, with a recorder script
   (committed) that captures real responses from OpenRouter and from Ollama
   running `qwen3:4b` (`ollama pull qwen3:4b`) into `tests/fixtures/`.
   Fixtures are committed after scrubbing auth headers, so the offline suite
   needs neither network nor Ollama.
3. Exercise the FastAPI app in-process (ASGI transport) instead of over
   localhost.
4. Add a `live` marker (OpenRouter, one cheap call per wire format) and an
   `integration` marker; move the existing running-service tests under
   `integration`.
5. Live tests append each call's cost (from OpenRouter's usage accounting)
   to `tests/live/spend.jsonl`, which is tracked and committed after every
   live run. The live-test guard sums that file and refuses to run past the
   $5 epic budget. Cross-check: the OpenRouter key showed $2.90 cumulative
   usage on 2026-10-26, before any epic spend.
6. Isolate tests from machine state: the offline suite must not read
   `~/.model_calling/capability_cache.json`, `.env`, or the generated
   `tests/models_available.py`. `conftest.py` points the capability cache at
   `tmp_path` and skips dotenv loading for offline tests.
7. Add a grid helper that parametrizes tests over (ingress, backend) pairs,
   so later phases add cells rather than ad hoc tests.
8. CI runs the offline suite on 3.11 through 3.14.

**Definition of done:** plain `pytest` passes offline in under 30s with no
network access, in a fresh clone with an empty home directory
(`HOME=$(mktemp -d) pytest`). `pytest -m live` passes against OpenRouter and
the spend ledger is committed. `gh run list` shows the offline suite green on
CI.

**Dependencies:** none.

**Parallel-ok:** no. Everything else depends on it.

### Phase 2: Neutral internal format and adapter port

Replace "OpenAI Chat Completions dict as the hub format" with a typed,
block-based internal format, so each ingress and backend is one translator
instead of N×M.

**Work:**
1. Define typed internal request/response/stream-event models (pydantic).
   A turn is a list of typed content blocks (text, image, tool call, tool
   result, thinking with signature), with per-block metadata such as
   `cache_control`. Do not model it on Chat Completions: flat `content`,
   message-level `tool_calls`, and the `tool` role are translation concerns
   for that one format. Round-tripping Anthropic → internal → Anthropic must
   be lossless.
2. Define the provider interface: translate request, translate response,
   translate stream events, plus endpoint and auth. Streaming goes through
   the adapter; delete the bypass in `server.stream_response`.
3. Port Ollama, vLLM, and OpenAI-compatible backends. Add OpenRouter as a
   named provider.
4. Fix the bugs found during cleanup, each with a regression test: Ollama
   `call_model` skips callbacks; unprefixed model names hijack
   `/v1/moderations`; backend errors return HTTP 200; library modules call
   `logging.basicConfig`; response IDs built from `int(time.time())`
   collide; hosted adapters crash every non-streaming call because
   callbacks receive `**translated_request`, whose `model` key collides
   with the positional model name (found by the Phase 1 grid);
   `stream_response` appends `/v1/chat/completions` to hosted adapters'
   full endpoint URL and sends no auth header. The strict xfails in
   `tests/harness/grid.py` track these.
5. Remove the keyword-based "reasoning" warning heuristic in the server.

**Definition of done:** fixture round trips pass for Ollama, vLLM, OpenAI,
and OpenRouter through the new interface. A test round-trips an Anthropic
message with thinking, tool use, and `cache_control` through the internal
format with no loss. Each listed bug has a test that failed before the fix.
`model_calling/server.py` makes no direct backend HTTP calls.

**Dependencies:** Phase 1.

**Parallel-ok:** no.

### Phase 3: Library API with both formats, server as a thin layer

The near-term use case: import model-calling in place of LiteLLM, with no
server running. Familiar to LiteLLM users where that costs nothing, better
where it matters. rdwj confirmed these design proposals on 2026-10-28:

- **Explicit client, no global state.** `Client(...)` owns config,
  callbacks, and a pooled httpx client. Module-level functions are
  conveniences over a default client.
- **Both formats first-class.** `completion` / `acompletion` return OpenAI
  types; `messages` / `amessages` return Anthropic types. Each supports
  streaming. `responses` follows in Phase 7.
- **Strict by default.** Parameters a backend cannot honor raise an error
  naming the parameter; dropping is an explicit opt-in.
- **Typed responses and errors.** Pydantic response models that validate
  against the SDK types; exceptions carry provider, status, and raw body.

**Work:**
1. Public API: `Client` plus module-level `completion`, `acompletion`,
   `messages`, `amessages`, `embedding`, `aembedding`, with `provider/model`
   routing and per-call `api_base`/`api_key` overrides.
2. Typed response models: Chat Completions output validates against
   `openai.types.chat.ChatCompletion` / `ChatCompletionChunk`; Messages
   output validates against `anthropic.types.Message` and the stream event
   types.
3. Callback registration on the client, opt-in parameter dropping, and an
   exception hierarchy with OpenAI-style names (`AuthenticationError`,
   `RateLimitError`, `BadRequestError`, and so on) mapped from backend
   errors.
4. Move FastAPI, uvicorn, and python-dotenv into a `[server]` extra, still
   exact-pinned; the core library depends only on httpx and pydantic and
   never reads `.env`. The server exposes both
   `POST /v1/chat/completions` and `POST /v1/messages` over the same core.
5. Replace the HTTP-only `ModelCallingClient` docs with in-process usage as
   the primary path.

**Definition of done:** a clean `pip install model-calling` (no extras) can
run `acompletion` and `amessages` against a fixture backend.
`pip install model-calling` does not pull in fastapi. The same calls through
the server produce identical bodies. Both response families pass SDK type
validation for a basic text exchange.

**Dependencies:** Phase 2.

**Parallel-ok:** with Phase 6.

### Phase 4: Chat Completions conformance

**Work:**
1. Request model accepts current fields: content-part arrays (text, image),
   `tool_choice`, `parallel_tool_calls`, `max_completion_tokens`,
   `response_format` with `json_schema`, `stream_options.include_usage`,
   `reasoning_effort`, `seed`, parameterless tools, `strict`.
2. Responses emit `tool_calls` (all of them), never legacy `function_call`.
3. Fill the Chat Completions row of the grid: text, parallel tools, image
   input, `json_schema`, and streaming with usage, through the library and
   through `openai.OpenAI(base_url=...)` against the server.

**Definition of done:** the Chat Completions row passes offline for every
ported backend and live via OpenRouter.

**Dependencies:** Phase 3.

**Parallel-ok:** with Phases 5 and 6.

### Phase 5: Anthropic Messages conformance

The Messages ingress against non-Anthropic backends (Ollama, vLLM,
OpenAI-compatible). It does not need the Anthropic backend.

**Work:**
1. Request model accepts current Messages fields: system as string or
   blocks, image blocks, `tools` and `tool_choice`, `tool_result` with
   `is_error`, `thinking` config, `stop_sequences`, `metadata`, and
   `cache_control` (pass through where the backend supports it, otherwise
   ignore without error).
2. `POST /v1/messages/count_tokens` (estimate for backends without a
   native counter, documented as such).
3. SSE output emits the full event sequence (`message_start`,
   `content_block_start`, `content_block_delta` including
   `input_json_delta`, `content_block_stop`, `message_delta`,
   `message_stop`).
4. Fill the Messages row of the grid with the same scenarios as Phase 4,
   through the library and through `anthropic.Anthropic(base_url=...)`
   against the server.

**Definition of done:** the Messages row passes offline for every ported
backend and live via OpenRouter's Chat Completions backend, which exercises
Messages → Chat Completions translation.

**Dependencies:** Phase 3.

**Parallel-ok:** with Phases 4 and 6.

### Phase 6: Anthropic and Cohere v2 backends

**Work:**
1. Rewrite the Anthropic adapter from the current Messages API spec:
   `tool_use`/`tool_result` blocks, multiple system blocks, stop reason and
   usage mapping, SSE event parsing, images, extended thinking,
   `cache_control` pass-through. Drop the hardcoded Claude 3 model tables.
2. Rewrite the Cohere chat and embedding adapters against the v2 API.
3. Point the Anthropic backend at OpenRouter `/api/v1/messages` for live
   tests (confirm the auth header it expects). vLLM also serves
   `/v1/messages` natively (checked 2026-10-28), so the same provider
   should run against vLLM; its `vllm_messages` fixtures (Phase 3 slice 1)
   give the column a free, offline second backend.
4. Add the Anthropic and Cohere columns to the grid for every ingress that
   exists so far.

**Definition of done:** the Anthropic column passes offline and live via
OpenRouter for text, tools, streaming, and thinking. Messages in →
Anthropic out is a lossless pass-through in tests. Cohere v2 passes fixture
checks. There is no Cohere key, so Cohere has no live coverage; the README
says so.

**Dependencies:** Phase 2.

**Parallel-ok:** with Phases 3, 4, and 5.

### Phase 7: OpenAI Responses ingress and backend (stateless)

**Work:**
1. `POST /v1/responses`, streaming and non-streaming, plus `responses` /
   `aresponses` in the library.
2. A Responses backend adapter (OpenRouter `/api/v1/responses` for live
   tests).
3. `previous_response_id`, `store: true`, and other state-dependent features
   return a clear 400 that names the unsupported field.
4. Fill the Responses row of the grid.

**Definition of done:** `client.responses.create(...)`, with and without
streaming, passes the Responses row offline and live via OpenRouter.
Stateful requests get the documented 400.

**Dependencies:** Phase 4.

**Parallel-ok:** with Phases 5, 6, and 8.

### Phase 8: Claude Code acceptance

The headline end-to-end test of the Messages path.

**Work:**
1. Acceptance script: start the server, set `ANTHROPIC_BASE_URL` to it, and
   run `claude -p` on a scripted tool-using task in a temp directory (create
   a file, edit it, run a command). Check the resulting files
   deterministically.
2. Run it against (a) a Claude model via the Anthropic backend on
   OpenRouter, (b) a non-Claude model via OpenRouter Chat Completions, and
   (c) Ollama running `qwen3-coder:30b` (`ollama pull qwen3-coder:30b`;
   fallback `gpt-oss:20b`).
3. Commit the acceptance script under `tests/acceptance/` with a README
   covering prerequisites: Claude Code installed, `OPENROUTER_API_KEY` in the
   environment, and the Ollama model pulled.

**Definition of done:** the script passes for all three backends, and the
run's spend is committed to `tests/live/spend.jsonl`.

**Dependencies:** Phases 5 and 6.

**Parallel-ok:** with Phase 7.

### Phase 9: Pass-through, release, and pilot migration

**Work:**
1. Pass-through mode: when the backend natively speaks the ingress format
   (for example vLLM or OpenRouter Messages/Responses), forward without
   round-tripping through the internal format. vLLM's native Messages
   endpoint is confirmed (2026-10-28); verify Ollama's before relying on it.
2. Switch PyPI publishing to trusted publishing (OIDC) and drop the token
   secret.
3. Write a "Migrating from LiteLLM" doc in `docs/` (nominative mention
   only) that lists every intentional difference.
   Breaking changes collected so far (Phase 2): vLLM's legacy `/generate`
   prompt mode and the `openai_compatible` switch are gone; `base_url` for
   `vllm`, `openai`, and `openrouter` is the API root with `/v1` included,
   as in the `openai` SDK (was the server root for vLLM and the full
   `/chat/completions` URL for OpenAI); a backend's 401/403 reaches the
   client as 502.
4. Tag and release 0.2.0.
5. Pilot migration: move `src/sanicode/llm/client.py` in
   [rdwj/sanicode](https://github.com/rdwj/sanicode) (private) off LiteLLM,
   and run sanicode's tests. If memory-hub's LiteLLM question (an MCP tool
   idea from one of its maintainers) has surfaced by then, consider the
   hand-rolled `/v1/chat/completions` calls in
   [redhat-ai-americas/memory-hub](https://github.com/redhat-ai-americas/memory-hub)
   (`src/memoryhub_core/services/extraction.py` and
   `src/memoryhub_core/services/dreaming.py`) as a second pilot.

**Definition of done:** `pip install model-calling==0.2.0` in a fresh 3.11
venv runs an `acompletion` and an `amessages` smoke call. sanicode's test
suite passes with LiteLLM removed from its dependencies.

**Dependencies:** Phases 7 and 8.

**Parallel-ok:** no.

---

## What this covers (and what it doesn't)

**In scope:**
- OpenAI Chat Completions and Anthropic Messages as co-equal ingress formats
- OpenAI Responses (stateless), ingress and backend
- Anthropic Messages and Cohere v2 backends
- OpenRouter as a first-class provider
- Library API familiar to LiteLLM users, with documented breaking changes
- Bugs surfaced during the public-release cleanup
- 0.2.0 release and a pilot migration

**Out of scope (future epics):**
- Local Responses store (`previous_response_id`, `store`, retrieval endpoints)
- Token counting and cost accounting (`token_counter`, `cost_per_token`,
  pricing data)
- Router features (fallbacks, load balancing, retries across deployments)
- Granite Guardian / moderation rework
- FIPS container validation (`fips-todo.md`)

## What landed last session (2026-10-28): Phase 2 slice 2, Phase 2 complete

All four slice 2 items plus review fixes, `fd2679c..2847c81`. One
OpenAI-compatible provider serves `vllm/`, `openai/`, and `openrouter/`.
`server.py` makes no HTTP calls. `KNOWN_FAILURES` is empty, and every
remaining Phase 2 bug has a regression test. No OpenRouter spend; vLLM
fixtures were recorded from rdwj's gpt-oss-20b. Full record, including
judgment calls and the grid's known blind spots:
`session-summaries/2026-10-28-api-shapes-phase2-slice2.md`.

Still open from earlier slices:
- Document and Refusal block types (Phases 5 and 7).
- `reasoning_effort` strings for Ollama `think` (Phase 4).
- Anthropic egress of thinking with no signature (now item 3 above).
- Responses stream structure (Phase 7).
- `on_stream_chunk` receiving a raw str (Phase 3, with the callback API).
- `/v1/models/refresh` returns 404 for `vllm/` models since the legacy
  adapter was deleted (fold into slice 2's server work).

## What landed earlier (2026-10-28): Phase 2 slice 1

The internal format (`model_calling/core/`), lossless Anthropic round
trips, Chat Completions ingress, and Ollama on the provider interface,
`b067f38..7e4e44c`. Record:
`session-summaries/2026-10-28-api-shapes-phase2-slice1.md`.

## What landed earlier (2026-10-26, evening): Phase 1 complete

Phase 1's definition of done is met on `8b6120e`. The offline suite passes
in about 1s with an empty home directory, `pytest -m live` passes on all
three wire formats, and every `Test Package` job is green. Session spend:
$0.000007 across 18 ledger entries (cumulative epic total is the same).
The OpenRouter key showed $3.15 used before any epic spend, not the $2.90
noted earlier. Full record: `session-summaries/2026-10-26-api-shapes-phase1-harness.md`.

The grid's first row has 1 passing cell and 7 strict xfails, which make up
Phase 2's starting bug list, plus one Phase 4 item. `anthropic` is pinned
at 1.15.0 because 2.0.0 (2026-10-21) postdates the cutoff; moving the cutoff
on or after 2026-10-28 15:51Z makes 2.x eligible (decided: move it, see
item 0 above).

## What landed earlier (2026-10-26)

Public-release prep and planning. Relicensed to Apache-2.0, moved to PEP 621
packaging with a Python 3.11 floor, lazy-loaded the torch-dependent
moderation adapter, removed docs for unbuilt companion projects, rewrote
history to a single root commit, and made the repo public. Planned this epic
and adopted the dependency policy.

**Commits:** `23f63dc` initial public release; `27866ab` independent-
implementation and offline-testing rules; `2f9a2a8` epic plan; `1fdd8c0` exact
pins behind a 7-day cutoff, SHA-pinned Actions, digest-pinned images, and the
CI dependency gate (the first audit caught certifi PYSEC-2026-4320);
`2edf055` dependency policy docs; `5d76b5b` policy applied to the phases.

**CI baseline on `5d76b5b`:** dependencies job green; lint red on two F821
bugs; tests red because they need a running service. Black would reformat 22
of 23 files.

## Watch out for

- **Dependency policy.** Every external dependency is pinned exactly (`==`,
  commit SHA, or image digest), at least 7 days old, and free of known major
  CVEs. `[tool.uv] exclude-newer` and the CI `pip-audit` job enforce it.
  LiteLLM must never appear in `uv.lock`. Any phase that adds a dependency
  follows "Dependency Policy" in `CONTRIBUTING.md`.
- **Clean-room rule.** Where the API resembles LiteLLM, it matches names and
  call patterns only, built from the
  OpenAI, Anthropic, Cohere, and OpenRouter public specs and SDK types. Do
  not read, copy, or adapt LiteLLM source, docstrings, docs, type
  definitions, or data files (such as its model pricing JSON). Never read its
  separately licensed `enterprise/` directory. Mention LiteLLM only in the
  migration guide.
- **SDK versions.** The repo pins `openai` 3.35.1 and `anthropic` 2.0.0
  (cutoff 2026-10-21T15:52Z). Check current types in the installed SDK
  source before writing translators; do not rely on memory. The SDK's
  request TypedDicts validate lazily and ignore unknown keys; use
  `validate_anthropic_request`, which forces deep validation.
- **The harness is the exit predicate.** New offline coverage goes in the
  grid (`tests/harness/grid.py`, `tests/test_gateway_grid.py`), with
  fixtures from `uv run python -m tests.harness.record`. See
  `tests/README.md`.
- **Live budget: $5 for the whole epic** on OpenRouter (the key has a $50
  cap), enforced from the committed `tests/live/spend.jsonl`. Use cheap
  models and keep `-m live` out of the default run. Live tests need
  `OPENROUTER_API_KEY` in the environment, never in a file.
- **Fixtures must be scrubbed** of auth headers and keys before commit; run
  gitleaks on `tests/fixtures/`.
- The repo is public. Planning files, fixtures, and the spend ledger are
  visible.
- **No local-only dependencies.** Everything the plan or the tests rely on
  is committed, fetched from a public source (PyPI, the Ollama library), or
  read from an environment variable. The clean-room rule and the budget are
  also recorded in `CONTRIBUTING.md` and `CLAUDE.md`.

## If blocked

- If the stricter replay (item 0) turns up a real gateway bug that takes
  more than a small fix, record it as a strict xfail naming Phase 3 and
  move on to item 1.
- If the Anthropic SDK's stream event types make the encoder's shape
  ambiguous (for example thinking without a signature), finish
  `completion`/`acompletion` with streaming first and bring the question
  to rdwj with the concrete event sequence.
- If an OpenRouter endpoint diverges from the native provider API (auth
  header, beta fields), record it in the fixture README and test the native
  shape from SDK types offline.
