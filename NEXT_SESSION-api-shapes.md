# Next Session — api-shapes

## Next: Phase 5, Anthropic Messages conformance

Prove the Messages ingress end to end over Ollama, vLLM, and OpenRouter for
every Messages feature a client sends, not just text and one tool. Phase 3
already built `POST /v1/messages`, the full SSE event sequence, and a
4-scenario `gateway_messages` row; the translator in
`model_calling/formats/anthropic.py` already parses system blocks, images,
`tool_choice`, `tool_result` with `is_error`, `thinking`, and
`stop_sequences` (checked 2026-10-28). What's missing is coverage per
backend, SDK-client cells, `count_tokens`, and typed-block input.

1. **New grid scenarios** (record, then cells for `library_messages`,
   `gateway_messages`, and the Chat Completions ingresses where the
   scenario applies). In `tests/harness/scenarios.py`: system as a list of
   blocks; a multi-turn tool exchange whose `tool_result` has
   `is_error: true`; parallel tool calls (two cities); image input (base64
   PNG, committed small; needs a vision model per backend: Ollama
   `gemma3`/`llava:7b` are pulled, a cheap OpenRouter vision model, vLLM's
   model if it has vision, otherwise a strict xfail naming the reason);
   `thinking` enabled; streaming with usage. Scenarios may need fields
   `Scenario` doesn't have yet (system, prior turns, image); extend it
   rather than special-casing the recorder.
2. **SDK-client cells.** A grid ingress that drives the in-process server
   with `anthropic.Anthropic(base_url=..., http_client=<ASGI transport>)`
   for every Messages scenario, streaming through the SDK's own stream
   helper. This is the client Claude Code-style apps use.
3. **`POST /v1/messages/count_tokens`.** No backend here has a native
   counter; return an estimate and document it as one (README,
   api-reference). Validate the body against the SDK's
   `MessageTokensCount`.
4. **Typed blocks as input.** `messages()` accepts the `Message.content`
   it returned (pydantic blocks) as the next assistant turn, as the
   `anthropic` SDK does. Then simplify the `.to_dict()` workaround in
   `docs/examples.md`.
5. **Live check through our code.** A `-m live` test runs `amessages`
   (text and tools) through `Client` over OpenRouter, and through the
   gateway; spend goes to the ledger. Phase 5's definition of done needs
   it.

**Sequencing.** 1 first (fixtures gate everything else), recording Ollama
and OpenRouter early and vLLM as soon as its settings are confirmed. Then 2,
which reuses the new scenarios. 3 and 4 are independent and small. 5 last.
If the session runs long, stop after 4 with 5 written up; don't cut 2.

**Constraints for the session:**
- Exit predicate: `HOME=$(mktemp -d) uv run pytest` green, with every new
  scenario present in the Messages rows for ollama, vllm, and
  openai_compat (passing, or a strict xfail naming the backend limitation
  and phase); the existing cells unchanged; `uv run pytest -m live`
  passing with the new spend committed to `tests/live/spend.jsonl`.
- vLLM: record against the cluster rdwj is using now. The recorder reads
  `VLLM_BASE_URL` (server root, no `/v1`), `VLLM_MODEL`, optional
  `VLLM_API_KEY` and `VLLM_VERIFY_TLS` from the environment; fixtures store
  the replay URL and `gpt-oss-20b`, never the real host.
- OpenRouter budget: the full $5 epic budget (decided 2026-10-28), enforced
  by the ledger guard; about $0.00001 was spent before this session. Log
  every paid call; prefer cheap models.
- Phase 5's definition of done (Remaining epic phases) should be fully met;
  say plainly in the session summary if any clause isn't.

**Session start protocol:**
- Premise checks (before item 1, report before acting): `main` matches
  `origin/main` at the plan commit after `c7d6638`; "Test Package" is green;
  `HOME=$(mktemp -d) uv run pytest` shows 541 passed, 1 xfailed;
  `VLLM_BASE_URL` and `VLLM_MODEL` are set in the environment and
  `curl $VLLM_BASE_URL/v1/models` answers (if not, ask rdwj for the current
  cluster's values before recording vLLM); Ollama is up with `qwen3:4b`
  and a vision model.
- Rules with history:
  - Never loosen a grid check or replay matching to make a cell pass;
    re-record or strict-xfail naming the phase. Record OpenRouter
    divergences in `tests/fixtures/README.md`.
  - Read sub-agent output before committing, and run examples against
    fixtures: last session's doc writers used `content[0].text` on
    thinking-first responses.
  - Gateway behavior changes only where this plan says so; diff the
    gateway grid's bodies, not just pass/fail.
- Stop-and-ask before: moving the dependency cutoff or changing a pinned
  version; any change to the vLLM deployment; removing or renaming a
  public endpoint; force-pushing.
- Close ritual: `/session-close`, then `/plan-next-session`. Next pick is
  among Phases 6, 10, and 4.

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

Phases 1–3 are done (harness, internal format and adapter port, library API
with the server on `Client`); see the "What landed" ledger and
`session-summaries/`. Phase numbers are kept stable because code comments,
xfail reasons, and summaries cite them, so the gateway-configuration phase
added on 2026-10-28 is Phase 10 and sits before Phase 9, which it gates.
Order: Phase 5 next (decided 2026-10-28), with 6 and 10 parallel-ok, then
4; 7 after 4; 8 after 5 and 6; 9 last.

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
OpenAI-compatible). It does not need the Anthropic backend. Phase 3 already
built `POST /v1/messages`, the SSE encoder, and a `gateway_messages` grid
row (text, tools, stream, backend_error) that is compared cell by cell with
the library, so this phase adds fields, scenarios, and SDK-client cells.

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
5. `messages()` accepts the typed response blocks it returns as the next
   assistant turn (today they must go through `.to_dict()`; the
   `anthropic` SDK accepts its own blocks there). Found 2026-10-28.

**Definition of done:** the Messages row passes offline for every ported
backend and live via OpenRouter's Chat Completions backend, which exercises
Messages → Chat Completions translation. A grid cell drives the server with
`anthropic.Anthropic(base_url=...)` for every Messages scenario, and a
library test feeds a typed `Message.content` straight back into
`messages()`.

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

### Phase 10: Gateway configuration and hardening

Make the gateway deployable outside localhost and safe to run on a shared
network. Found during Phase 3 slice 2: the container files set backend URLs
the gateway never applies, and the runtime config endpoint can redirect the
gateway's API keys.

**Work:**
1. Apply `OLLAMA_BASE_URL` and `VLLM_BASE_URL` (and a vLLM key variable) to
   the gateway's provider configs. Decide at the start whether
   `VLLM_BASE_URL` is the server root or the API root with `/v1`
   (`Containerfile`, `podman-compose.yml`, and the recorder use the server
   root; the provider and library use the API root). Remove the logging-only
   `config.load_config` path or make it the one source.
2. Protect `POST /v1/config/update`: off by default with an environment
   switch, or a shared-secret header. Changing `base_url` must not be able
   to send a hosted key to an arbitrary host unauthenticated.
3. Stop `server.py` from loading `.env` at import: load it in the entry
   point (`__main__`) so importing the app in tests or other processes
   never picks up real keys.
4. Document the settings in README and `docs/advanced.md`, replacing the
   "not applied yet" notes.

**Definition of done:** a gateway test sets `OLLAMA_BASE_URL` and
`VLLM_BASE_URL` and the replayed requests go to those hosts;
`podman-compose up` reaches the Ollama container with no
`/v1/config/update` call; an unauthenticated `/v1/config/update` gets 403
(or 404 when disabled); `python -c "import model_calling.server"` in a
directory with a `.env` leaves `os.environ` unchanged; no doc says "not
applied yet".

**Dependencies:** Phase 3.

**Parallel-ok:** with Phases 4, 5, and 6. Must land before Phase 9 (the
pilot migration runs the gateway against a configured backend).

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
   client as 502. Phase 3: the gateway needs `model-calling[server]`;
   the library is strict by default (`drop_params=True` opts out); an
   early-stopped stream ends with `on_request_end(cancelled=True)`, not
   `on_error`; `on_stream_chunk` receives internal events, not raw lines;
   `/v1/embeddings` returns the client's model id and the backend's token
   count, and its cache covers only Cohere; gateway bodies may not carry
   `api_base`, `api_key`, or `drop_params`; errors a backend reports inside
   a stream map by kind (429, 404, 400) instead of always 502; gateway JSON
   key order follows the SDK types. `docs/migration-guide.md` has an
   "Upgrading from 0.1.x" section to fold into the LiteLLM doc's sibling.
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

**Dependencies:** Phases 7, 8, and 10.

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
- Gateway configuration from the environment and a protected config endpoint
- 0.2.0 release and a pilot migration

**Out of scope (future epics):**
- Local Responses store (`previous_response_id`, `store`, retrieval endpoints)
- Token counting and cost accounting (`token_counter`, `cost_per_token`,
  pricing data)
- Router features (fallbacks, load balancing, retries across deployments)
- Granite Guardian / moderation rework
- FIPS container validation (`fips-todo.md`)

## What landed last session (2026-10-28): Phase 3 slice 2, Phase 3 complete

All five items plus review fixes, `b8522a3..c7d6638`: server over `Client`,
`POST /v1/messages`, embeddings, strict mode on the server, the `[server]`
extra, and docs led by in-process use. Phase 3's definition of done is met.
Record: `session-summaries/2026-10-28-api-shapes-phase3-slice2.md`.

## What landed earlier (2026-10-28): Phase 3 slice 1

All six items plus review fixes, `5e506df..4fadc21`: strict replay (it
found two more blind spots), `vllm_messages` reference recordings, the
exception hierarchy, `Client` with `completion`/`messages` sync and async
with streaming, the Anthropic stream encoder, and strict-by-default with
`drop_params`. 24 library grid cells pass; 425 tests. OpenRouter spend
$0.000002 (re-recording). Full record:
`session-summaries/2026-10-28-api-shapes-phase3-slice1.md`.

Still open from earlier slices:
- Document and Refusal block types (Phases 5 and 7).
- `reasoning_effort` strings for Ollama `think` (Phase 4).
- Responses stream structure (Phase 7).
- The Phase 6 Anthropic provider must drop unsigned thinking
  (`signature: ""`) before sending it to Anthropic.

## What landed earlier (2026-10-28): Phase 2 slice 2

One OpenAI-compatible provider for `vllm/`, `openai/`, `openrouter/`;
`fd2679c..2847c81`. Record:
`session-summaries/2026-10-28-api-shapes-phase2-slice2.md`.

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
  grid (`tests/harness/grid.py`, `tests/test_gateway_grid.py`,
  `tests/test_library_grid.py`), with
  fixtures from `uv run python -m tests.harness.record`. See
  `tests/README.md`.
- **Live budget: $5 for the whole epic** on OpenRouter (the key has a $50
  cap), enforced from the committed `tests/live/spend.jsonl`. Use cheap
  models and keep `-m live` out of the default run. Live tests need
  `OPENROUTER_API_KEY` in the environment, never in a file.
- **Fixtures must be scrubbed** of auth headers and keys before commit; run
  gitleaks on `tests/fixtures/`.
- **Sync API shutdown is delicate.** `model_calling/library/_sync.py`
  cancels in-flight calls and closes streams before stopping its loop;
  review found three hangs there. Keep the regression tests in
  `tests/library/test_client.py` green when touching `Client.close`.
- The repo is public. Planning files, fixtures, and the spend ledger are
  visible.
- **No local-only dependencies.** Everything the plan or the tests rely on
  is committed, fetched from a public source (PyPI, the Ollama library), or
  read from an environment variable. The clean-room rule and the budget are
  also recorded in `CONTRIBUTING.md` and `CLAUDE.md`.

## If blocked

- If vLLM settings aren't available, record Ollama and OpenRouter, add the
  vLLM cells for new scenarios as strict xfails ("needs recording, Phase
  5"), and carry the recording into the next session.
- If a backend can't express a scenario (no vision model on vLLM), make it
  a strict xfail naming the limitation instead of dropping the cell.
- Phase 10 (gateway configuration) is parallel-ok and small; it's the
  fallback if Phase 5 stalls on recordings.
