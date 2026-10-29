# Next Session — api-shapes

## Next: Phase 11 slice 1, model aliases from a hot-reloaded file

Gateway clients ask for an alias (`chat`, `embed`) and a YAML file decides
which backend and model serve it, reloaded without a restart. This is the
project's original purpose: the app never changes when Ollama in dev
becomes vLLM in production. rdwj made the design calls on 2026-10-29; they
are listed at the top of Phase 11 below and are not open.

1. **The file and its loader.** Backends (provider, `base_url`,
   `api_key_env`) and models (backend, model, optional `dimensions` for
   embeddings), validated with errors that name the offending key. Start
   from the shape in Phase 11. Decide and record: the env variable for the
   path (suggest `MODEL_CALLING_CONFIG`) and whether the entry point
   defaults to `./model-calling.yaml`; what a gateway with no file does
   (suggest: starts, lists no models, and every request gets a 404 whose
   message names the variable). Pin PyYAML `==6.0.3` as a direct `[server]`
   dependency (already in the lock; check `pip-audit`).
2. **Hot reload.** `os.stat` per request (follows symlinks, so a ConfigMap
   `..data` swap counts); compare mtime and size; parse and validate the
   new file before swapping it in; an invalid edit keeps the last good
   config and logs one error per change, not per request. Each request
   resolves its alias once, at the start.
3. **Alias routing for Chat Completions and Messages.** Resolve the alias to
   a provider config and model, call `Client` with that, and put the alias
   in the response `model` field (streaming too). `/v1/models` lists
   aliases only. An unknown name, including any `prefix/model`, is a 404 in
   the ingress's error format naming the alias. Phase 10's URL variables
   remain defaults for a backend the file gives no URL.
4. **Embeddings.** Same routing for `/v1/embeddings`; with `dimensions`
   declared, a response of another size is a 502 naming the alias and both
   sizes.
5. **Move the grid to aliases.** The gateway fixture writes an alias file
   per cell and gateway cells send the alias. The library cells keep
   prefixed names (the library still takes them); where library and
   gateway output are compared, `model` differs by design.
6. **Prove the swap.** A test rewrites the file from an Ollama backend to a
   vLLM backend between two requests for `chat` and sees the second go to
   the vLLM fixture; a malformed rewrite leaves the old mapping serving.

**Sequencing.** 1 and 2 first with their own unit tests, then 3, then 5
(the grid will show what 3 broke), then 4 and 6. If the session runs long,
stop after 5 with the grid green; 4 and the dimensions check can move to
slice 2. Leave the legacy registry, `/v1/config/update`,
`/v1/models/refresh`, `/v1/moderations`, `Client(models=...)`, and the
compose/docs rewrite to slice 2, but add a short "Model aliases" section to
`docs/advanced.md` so the new behavior isn't undocumented.

**Constraints for the session:**
- Exit predicate: the swap test and malformed-rewrite test in item 6 pass;
  `prefix/model` and unknown names get 404 in both Chat Completions and
  Messages formats; `/v1/models` lists only aliases; the grid is green with
  no new xfails; `HOME=$(mktemp -d) uv run pytest` green; each commit
  passes on its own.
- Breaking for gateway clients: prefixed names stop working, and Cohere
  stops being reachable through the gateway until Phase 6 slice 2 ports it
  (decided: aliases target ported providers only). Add both to the
  migration guide's upgrading section in the same commit as the behavior.
- No live spend; offline only.

**Session start protocol:**
- Premise checks (before item 1, report before acting): `main` matches
  `origin/main` at the plan commit after `1741f28`; "Test Package" green on
  it; `HOME=$(mktemp -d) uv run pytest` shows 1355 passed, 35 xfailed;
  `grep -rni "alias" model_calling` finds no alias feature (nobody started
  this in parallel); PyYAML 6.0.3 still in `uv.lock`.
- Rules with history:
  - Never loosen replay matching or a validator to get the grid green;
    re-record or strict-xfail with the fixing phase named.
  - Gateway behavior changes only where this plan says so. Slice 1 does not
    remove endpoints; the legacy registry stays for the endpoints that
    still use it.
  - Check that every commit passes on its own.
- Stop-and-ask before: removing or renaming any public endpoint (slice 2
  owns those); any dependency beyond PyYAML or moving the cutoff; changing
  stored fixture URLs; force-pushing.
- Close ritual: `/session-close`, then `/plan-next-session` (next pick is
  Phase 11 slice 2).

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
tool-using task through the gateway. Gateway clients name models only by
alias (`chat`, `embed`), never by backend: a hot-reloaded YAML file maps
each alias to a backend and model, so swapping Ollama for vLLM in
production is a file edit with no app change and no redeploy. This was the
project's original purpose (rdwj, 2026-10-29). The epic ends with 0.2.0 on
PyPI and a pilot project migrated.

Phases 1–5 are done (harness, internal format and adapter port,
library API with the server on `Client`, Chat Completions and Messages
conformance); see the "What landed" ledger and
`session-summaries/`. Phase numbers are kept stable because code comments,
xfail reasons, and summaries cite them, so the gateway-configuration phase
added on 2026-10-28 is Phase 10 and sits before Phase 9, which it gates,
and the model-alias phase added on 2026-10-29 is Phase 11. Phase 6 slice 1
(Anthropic) and Phase 10 are done. Order: Phase 11 next, slice 1 then
slice 2 (decided 2026-10-29); then among 8, 6 slice 2 (Cohere, hand-built
fixtures: no key), and 7; 9 last.

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

**Dependencies:** Phases 5, 6, and 11 (the run maps Claude Code's model
ids, including its background Haiku model, through aliases).

**Parallel-ok:** with Phase 7.

### Phase 11: Model aliases from a hot-reloaded config file

Gateway clients stop naming backends. Today the model string carries the
backend (`ollama/qwen3:4b` in dev, `vllm/gpt-oss-20b` in prod), so the app
changes when the backend does. After this phase the app asks for an alias
and a YAML file decides what serves it. Calls made with rdwj on
2026-10-29, not open: the file holds backends (URL and the *name* of the
env var with the key, never a key) as well as models; it reloads without a
restart; `ollama/`-style names leave the gateway wire entirely; embeddings
get aliases; `POST /v1/config/update` is removed; `Client` accepts the file
opt-in and keeps accepting prefixed names (the library is the layer allowed
to know backends); aliases target ported providers only, so Cohere aliases
come with Phase 6 slice 2.

A starting shape for the file, to settle in slice 1:

```yaml
backends:
  local: {provider: ollama, base_url: http://ollama:11434}
  prod: {provider: vllm, base_url: https://vllm.example/v1, api_key_env: VLLM_API_KEY}
models:
  chat: {backend: local, model: qwen3:4b}
  embed: {backend: local, model: nomic-embed-text, dimensions: 768}
```

Named backends allow two instances of one provider. The Phase 10 URL
variables stay as defaults for a provider the file names without a URL,
and the file wins where both set one.

**Slice 1: the file and alias routing.**
1. Loader with a schema (unknown keys, unknown backends, a key literal
   instead of `api_key_env`, and duplicate aliases are errors naming the
   line or key). Path from an env variable; decide the name and whether
   `./` gets a default.
2. Hot reload: check the file's mtime (following symlinks, so an OpenShift
   ConfigMap's `..data` swap counts) per request or on a short interval; an
   invalid edit keeps the last good config and logs the error; a request
   in flight finishes on the config it started with.
3. Chat Completions, Messages, and embeddings resolve the alias to a
   provider and model; the response `model` field and `/v1/models` show
   aliases only; an unknown name, including any `prefix/model` name, is a
   404 in the ingress's error format naming the alias.
4. An embedding alias with `dimensions` fails loudly (a 502 naming the
   alias and both sizes) when the backend returns another size, so a swap
   that would corrupt a vector index can't pass silently.
5. The grid moves to aliases: the harness writes an alias file per cell,
   gateway cells send aliases, and the library-vs-gateway comparisons
   account for the alias in `model`.

**Slice 2: removals, library, docs.**
1. Remove the gateway's legacy wire routing: the unprefixed-name default to
   Ollama and the fallthrough to `registry` adapters for chat and
   embeddings. Decide what `/v1/models/refresh` and `/v1/moderations` take
   as a model name, or retire them (stop-and-ask: public endpoints).
2. Remove `POST /v1/config/update`, `require_admin`, and
   `MODEL_CALLING_ADMIN_TOKEN` (stop-and-ask confirmed 2026-10-29), and the
   legacy client's `update_config`.
3. `Client(models="models.yaml")` with the same loader and reload; prefixed
   names still work in the library.
4. `podman-compose.yml` mounts a sample alias file; the docs show an
   OpenShift ConfigMap mount (not `subPath`, which never updates); the
   migration guide lists the removals and the prefix change; README leads
   the gateway section with the dev-to-prod swap.

**Definition of done:** a gateway test sends `model: chat` to a running
in-process gateway, rewrites the alias file from an Ollama backend to a vLLM
backend, and the next request goes to the vLLM fixture with no restart and
the response still says `chat`; a malformed rewrite leaves the old mapping
serving and logs the error; `prefix/model` and unknown names get 404 in both
Chat Completions and Messages formats; an embedding alias whose backend
returns the wrong size gets the documented 502; `POST /v1/config/update`
returns 404 and `grep -rn "config/update\|ADMIN_TOKEN" model_calling docs
README.md` finds nothing; `podman-compose up` serves `chat` from the Ollama
container using the mounted file, and editing the file on the host switches
the target without restarting the container; the grid stays green with no
new xfails.

**Dependencies:** Phase 10 (done).

**Parallel-ok:** with Phase 6 slice 2 (Cohere joins as an alias target when
it lands). Phases 7 and 8 should follow it: Responses would otherwise build
prefix routing that slice 2 removes, and Claude Code sends Claude model ids
that only aliases can map.

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

**Dependencies:** Phases 7, 8, 10, and 11.

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
- Model aliases from a hot-reloaded YAML file; no backend names on the gateway wire
- 0.2.0 release and a pilot migration

**Out of scope (future epics):**
- Local Responses store (`previous_response_id`, `store`, retrieval endpoints)
- Token counting and cost accounting (`token_counter`, `cost_per_token`,
  pricing data)
- Router features (fallbacks, load balancing, retries across deployments).
  Phase 11's aliases map one name to one target; they don't route.
- Granite Guardian / moderation rework
- FIPS container validation (`fips-todo.md`)

## What landed last session (2026-10-29): Phase 10 complete

All four items plus three review fixes, `47de0be..70e31fb`: backend URLs
and the vLLM key from the environment (`MODEL_CALLING_ANTHROPIC_BASE_URL`,
not `ANTHROPIC_BASE_URL`, by rdwj's call), the admin token on
`/v1/config/update` (slice 2 of Phase 11 removes the endpoint), `.env` read
only by the entry point from the start directory, and docs. Phase 10's
definition of done is met, including a `podman-compose up` check. Then
planned Phase 11 with rdwj. Record:
`session-summaries/2026-10-29-api-shapes-phase10.md`.

## What landed earlier (2026-10-28): Phase 6 slice 1

All four items plus review fixes, `c777d03..e155148`: the Anthropic
Messages provider on `anthropic/`, `vllm_messages` in the grid with
per-form fixtures and a pass-through test, typed `Tool.strict`, and a live
Anthropic column on OpenRouter (Haiku 5.5, Haiku 4.5 for thinking). Auth is
Bearer, as the anthropic 2.0.0 SDK sends it. Claude spend $0.0026. Record:
`session-summaries/2026-10-28-api-shapes-phase6-slice1.md`.

## What landed earlier (2026-10-28): Phase 4 complete

All four items plus review fixes, `5324bb1..fea1c2b`, and a sync-client
fix, `d85cdc3`. Phase 4's definition of done is met. Record:
`session-summaries/2026-10-28-api-shapes-phase4.md`. Two calls made with
rdwj: Ollama strict mode refuses `parallel_tool_calls: false` and strict
tools; Ollama looks up think levels via `/api/show` (`Provider.prepare`).

## What landed earlier (2026-10-28): Phase 5 complete

All five items plus `json_schema`, `stream_tools`, `output_config.effort`
and `.format` translation, and review fixes, `35e46c4..6dc8864`. Phase 5's
definition of done is met. Record:
`session-summaries/2026-10-28-api-shapes-phase5.md`.

## What landed earlier (2026-10-28): Phase 3 slice 2, Phase 3 complete

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
- Document and Refusal block types. Phase 5 did not do Document blocks
  (Messages `document` input arrives as an opaque block that every
  backend refuses); take it up with Phase 6 or as a small slice.
- No grid cell covers `Provider.prepare` on the streaming path, or the
  gateway's default drop path for refused Ollama cells (Phase 4 summary).
- Responses stream structure (Phase 7).
- From the Phase 6 slice 1 review (low): the Anthropic provider drops
  Chat Completions extras (`logprobs`, `logit_bias`, `user`) silently
  instead of refusing them in strict mode; streams lose `stop_details` and
  `container`; Chat-form `reasoning_effort` (adaptive thinking) is untested
  live against older Claude models; no cell pins `max_completion_tokens`
  to Messages `max_tokens`; `/v1/models` lists no `anthropic/` models.

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

- **The legacy registry probes every Ollama model at gateway startup.**
  With ~30 local models it saturates Ollama, which answers 503 "server
  busy" (the gateway returns 502) for a minute or two. Wait it out when
  driving a locally started gateway; Phase 11 slice 2 removes the legacy
  wire routing and is the place to retire it.

## If blocked

- If hot reload fights the test harness (timestamps with coarse mtime
  resolution), compare mtime, size, and inode, or let tests trigger a
  reload check directly; don't sleep in tests.
- If moving the grid to aliases balloons, give each cell a neutral alias
  generated from its (backend, scenario) in the fixture rather than
  hand-editing cells. Never alias a `prefix/model` name to itself: that
  reopens the wire to backend names.
- Fallback work in this epic: the Phase 6 slice 1 review leftovers listed
  under "Still open" (each small, offline).
