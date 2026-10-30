# Next Session — api-shapes

> **2026-10-29: cut for release.** 0.2.0 ships from `main` (library and
> gateway as they stand). The remaining work is tracked on GitHub:
> epic issue #8, issues #1–#7, milestone 0.3.0, project
> https://github.com/users/rdwj/projects/11. Phase-to-issue map: Phase 8 =
> #1, Phase 12 = #2, Phase 7 = #3 and #4, Phase 6 slice 2 = #5, Phase 9
> follow-ups = #6, pilot = #7. Update the issues as work lands; this file
> keeps the detailed arc. #1 (Phase 8) closed 2026-10-29; #2–#5 closed the
> same evening when PR #9 merged to `main` (`f9806bb`).
> 0.3.0 released to PyPI 2026-10-29 (`9c46d41`); the sanicode pilot is
> rdwj/sanicode#243, awaiting merge; friction filed as #10 and #11. Record:
> `session-summaries/2026-10-29-api-shapes-session-c.md`.

## Next: Release 0.3.0 and migrate sanicode off LiteLLM (#7)

The epic's end-state proof: sanicode's LLM client runs on model-calling
0.3.0 with LiteLLM gone from its dependencies and its test suite green.
The session starts by publishing 0.3.0 from Session B's merged work.

1. **Release 0.3.0** (needs rdwj at the publish step). PR #9 is already
   merged (`f9806bb`, closing #2–#5), so `main` holds everything 0.3.0
   ships. Bump `model_calling/__init__.py` `__version__` to `0.3.0`,
   check README line 20 and the migration guide name what 0.3.0 adds,
   then create the
   GitHub release `v0.3.0`; `.github/workflows/workflow.yml` publishes to
   PyPI on release (trusted publishing). Confirm `pip install
   model-calling==0.3.0` in a fresh 3.11 venv runs an `acompletion` and an
   `amessages` smoke call.

2. **#7: sanicode off LiteLLM** (rdwj/sanicode, private, checked out at
   `~/Developer/sanicode`; work on a branch there, never its `main`).
   LiteLLM lives in `src/sanicode/llm/client.py` (two `litellm.completion`
   calls, lines ~245 and ~369), `src/sanicode/llm/providers.py`
   (`ProviderInfo.litellm_prefix`), and tests `test_llm_client.py`,
   `test_providers.py`, `test_metrics.py` (they patch
   `sanicode.llm.client.litellm.completion`). Replace with
   `model_calling.completion` / a `Client`, rename `litellm_prefix` to the
   model-calling prefix, and swap `litellm>=1.83.0` for
   `model-calling>=0.3.0,<0.4` (sanicode's own range style). Decided with
   rdwj 2026-10-29:
   - gemini, mistral, groq, together_ai route through model-calling's
     `openai/` prefix at each one's OpenAI-compatible base URL;
   - azure and bedrock are dropped from sanicode's registry (api-version
     and SigV4 auth), with a note in sanicode's changelog/README;
   - anthropic, openai, cohere, ollama, vllm, openshift_ai map to native
     prefixes.
   Any model-calling friction found becomes a model-calling issue (and a
   0.3.x fix if small), recorded on #7.

**Sequencing.** 1 before 2: sanicode pins the published release.

**Constraints for the session:**
- sanicode tests: prefer driving a `model_calling.Client` with an
  `httpx.MockTransport` (as model-calling's own tests do) over patching
  `completion`; keep sanicode's existing test style where it has one.
- Clean room still applies in sanicode: replace LiteLLM calls from
  model-calling's API, never by reading LiteLLM internals.
- sanicode's README and config wizard mention LiteLLM model strings;
  update user-facing text along with the code.

**Session start protocol:**
- Premise checks (report before acting): `main` = `origin/main` at or
  after `f9806bb`, "Test Package" green on it; no `v0.3.0` tag or PyPI
  0.3.0 yet; offline suite on `main` shows 2494 passed,
  35 xfailed; sanicode clean on `main` at or after `f425bbd` (v0.12.3), and
  its suite green before any change (record the count).
- Rules with history:
  - Implementers this epic decided things their brief didn't cover (a
    spend cap, a backend refusal). Anything not decided above stops and
    asks.
  - sanicode lands by PR, never on its `main`; in model-calling, push the
    version bump to `main` only after rdwj says go.
- Stop-and-ask before: creating the `v0.3.0` release (publishes to PyPI;
  irreversible); dropping any sanicode provider beyond azure and bedrock;
  any change to sanicode's public CLI flags or config file format.
- Close ritual: session summary; record sanicode's before/after test
  counts; update #7 and #8; then `/plan-next-session` (left: #6, which
  needs a vLLM server, a remote amd64 build, and a cluster).

## Remaining epic phases

End-state: model-calling replaces LiteLLM in rdwj's projects and treats the
OpenAI and Anthropic wire formats as equals. The library API offers
`completion()` (OpenAI types) and `messages()` (Anthropic types) side by
side, familiar to LiteLLM users but not bound to its design; breaking
changes are documented in the migration guide. The optional gateway speaks
OpenAI Chat Completions, Anthropic Messages, and OpenAI Responses
(stateless) over Ollama, vLLM, OpenAI-compatible (including OpenRouter),
Anthropic, and Cohere v2. Gateway clients name models only by alias, from
a hot-reloaded YAML file, so a backend swap is a file edit. An offline
grid validates every ingress × backend cell against the official SDK
types; live smoke tests run through OpenRouter; Claude Code completes a
tool-using task through the gateway. 0.2.0 shipped the library and gateway
without Responses, Cohere, or the Claude Code proof; the epic ends with
0.3.0 and sanicode migrated off LiteLLM.

Done: Phases 1–8, 10, 11, and 12 (see the "What landed" ledger and
`session-summaries/`). Phase numbers stay stable because code comments,
xfail reasons, and summaries cite them, so the order below is not numeric.
Re-planned 2026-10-29 after the first reconciliation
(`reconciliations/2026-10-29-api-shapes.md`): Phase 9 split into two
slices, pass-through narrowed into Phase 7 slice 2, and the carried
offline leftovers given an owner as Phase 12.

**Session plan (decided 2026-10-29, after 0.2.0 shipped):**

| Session | Phases (issues) | Mode |
|---|---|---|
| A, evening of 2026-10-29 | 8 (#1) | Interactive; done |
| B, overnight after A | 12 (#2), 7 slices 1–2 (#3, #4), 6 slice 2 (#5) | Workflow, then finished interactively; done |
| C | 0.3.0 release, pilot (#7) | Interactive; the "Next" slice above |
| Later | 9 follow-ups (#6) | Needs a vLLM server, a remote build, and a cluster |

Order changed 2026-10-29: 0.3.0 releases before the pilot so sanicode pins
a published version; pilot friction ships as 0.3.x.

### Phase 8: Claude Code acceptance (done 2026-10-29)

`uv run pytest -m acceptance` passes on all three backends; see
`tests/acceptance/README.md` and
`session-summaries/2026-10-29-api-shapes-phase8.md`.

### Phase 12: Offline cleanup of carried leftovers

Items deferred across sessions without an owner. All offline, each small.

**Work:**
1. The Anthropic provider refuses `logprobs`, `logit_bias`, and `user` in
   strict mode, as it does `seed` and the penalties (decided 2026-10-29);
   the gateway still drops them by default.
2. Streams from the Anthropic provider carry `stop_details` and
   `container` through to Messages egress.
3. Grid cells for `Provider.prepare` on the streaming path and for the
   gateway's default drop path on a refused Ollama cell.
4. Messages `document` blocks: decide per backend (pass through to
   Anthropic and vLLM Messages; refuse with a named 400 elsewhere) and
   add the Refusal block type the SDK defines. Check the anthropic 2.0.0
   types first.
5. (Done in 0.2.0.) Delete `tests/integration/` and the root scripts that drive a running
   gateway with prefixed names (`run_tests.sh`, `test_ollama.sh`,
   `diagnose_models.sh`, `tool_call_curl.sh`, `auto_config_tests.sh`,
   `setup_tests.sh`, `fix_test_imports.sh`, `make_tests_executable.sh`,
   and any other that only serves them; check each before deleting) and
   the `integration` marker; update CONTRIBUTING (decided 2026-10-29).
6. A configurable request timeout (per prefix in `config` and per
   call); 0.2.0 fixes it at 300 s (`DEFAULT_TIMEOUT`).
7. What Phase 8 surfaced but didn't fix: the gateway drop-path cell for
   `thinking_unsupported` (with item 3), and a test pinning that strict
   mode stays silent on `context_management`, `metadata`, and
   `cache_control` for Chat Completions backends (decided 2026-10-29).

**Definition of done:** each item is a passing grid cell or test, or a
deletion; nothing is left in "Still open" or carried forward to a later
phase without a named owner;
`grep -rn "integration" pyproject.toml CONTRIBUTING.md` finds no test
marker or script; offline `pytest` green with no new xfails.

**Dependencies:** none.

**Parallel-ok:** with Phase 6 slice 2.

### Phase 7: OpenAI Responses ingress and backend (stateless)

Decided with rdwj 2026-10-29 for the first cut. **Stateless:** `store`
(absent or true) is accepted and the response reports `store: false`;
`previous_response_id`, `conversation`, and `background` get a 400 naming
the field. **Scope:** input message items (text, image),
`function_call`, and `function_call_output`; `instructions` becomes the
system prompt; output message and `function_call` items, plus reasoning
items carrying thinking as summary text. Built-in tools (`web_search`,
`file_search`, `code_interpreter`, computer use) and encrypted reasoning
(`include: reasoning.encrypted_content`) get a 400 naming them.

**Slice 1: ingress.**
1. `POST /v1/responses`, streaming and non-streaming, through
   `gateway._serve` with aliases from the start; `responses` /
   `aresponses` in the library.
2. The stateless and scope rules above, each with a grid cell.
3. A Responses grid row over the existing backends, including an
   `sdk_responses` ingress modeled on `sdk_completion`
   (`client.responses.create(...)` against the in-process gateway).

**Slice 2: backend.**
1. A Responses provider (OpenRouter `/api/v1/responses`) with recorded
   fixtures and a grid column.
2. Responses in → Responses backend is lossless in tests (this absorbs
   the old Phase 9 pass-through item; requests still go through the
   internal format, as Messages does).
3. Live smoke via OpenRouter.

**Definition of done:** `client.responses.create(...)`, with and without
streaming, passes the Responses row offline for every backend and live
via OpenRouter; stateful requests get the documented 400.

**Dependencies:** Phase 4 (done); slice 2 needs slice 1.

**Parallel-ok:** slice 1 with Phase 6 slice 2.

### Phase 6 slice 2: Cohere v2 backend

The 0.1.x Cohere adapters were deleted in Phase 11 slice 2, so this is a
new provider.

**Work:**
1. Cohere chat and embedding providers against the v2 API spec, on the
   provider interface.
2. Hand-built fixtures (no key) and a Cohere column in the grid for every
   ingress, including embeddings; Cohere usable as an alias target.
3. README and docs say Cohere has no live coverage.

**Definition of done:** the Cohere column passes fixture checks for text,
tools, streaming, and embeddings; an alias with `provider: cohere` loads
and serves in a gateway test.

**Dependencies:** Phase 2 (done).

**Parallel-ok:** with Phases 12 and 7.

### Phase 9: Release and pilot migration

Pulled forward into 0.2.0 on 2026-10-29: legacy client removed, PyPI page
now `README.md`, trusted publishing, live suite run on aliases. The rest
of slice 1 is #6; slice 2's pilot is #7, and its release becomes 0.3.0.

**Slice 1: release prep.**
1. Fold `docs/migration-guide.md` "Upgrading from 0.1.x" into a
   "Migrating from LiteLLM" doc (nominative mention only) listing every
   intentional difference. Two collected breaks aren't in the guide yet:
   gateway bodies may not carry `api_base`, `api_key`, or `drop_params`,
   and gateway JSON key order follows the SDK types.
2. Remove the legacy `ModelCallingClient` / `SyncModelCallingClient`
   (`model_calling/client.py`) with a migration-guide entry; CI's import
   check moves to `Client` (decided 2026-10-29).
3. Sync `PYPI_README.md` with `README.md`.
4. Switch PyPI publishing to trusted publishing (OIDC); drop the token.
5. Confirm vLLM's native `/v1/messages` against the `anthropic` provider.
6. Run the compose swap check on the amd64 image (remote build, ask
   first) and the live suite on aliases.

**Slice 2: release and pilot.**
1. Tag and release 0.2.0.
2. Move `src/sanicode/llm/client.py` in
   [rdwj/sanicode](https://github.com/rdwj/sanicode) (private) off
   LiteLLM and run its tests. If memory-hub's LiteLLM question has
   surfaced, consider its hand-rolled `/v1/chat/completions` calls
   (`src/memoryhub_core/services/extraction.py`, `dreaming.py` in
   [redhat-ai-americas/memory-hub](https://github.com/redhat-ai-americas/memory-hub))
   as a second pilot.

**Definition of done:** `pip install model-calling==0.2.0` in a fresh 3.11
venv runs an `acompletion` and an `amessages` smoke call; sanicode's test
suite passes with LiteLLM removed from its dependencies.

**Dependencies:** Phases 7, 8, 12, and 6 slice 2.

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
- Gateway configuration from the environment
- Model aliases from a hot-reloaded YAML file; no backend names on the gateway wire
- Removal of the legacy gateway client and the `prefix/model` gateway wire
- 0.2.0 release and a pilot migration

**Out of scope (future epics):**
- Local Responses store (`previous_response_id`, `store`, retrieval endpoints)
- Token counting and cost accounting (`token_counter`, `cost_per_token`,
  pricing data)
- Router features (fallbacks, load balancing, retries across deployments).
  Aliases map one name to one target; they don't route.
- A raw pass-through mode that skips the internal format
- Granite Guardian / moderation rework
- FIPS container validation (`fips-todo.md`)

## What landed last session (2026-10-29): Session B, Phases 6 slice 2, 7, and 12

All four streams merged onto `batch/0.3.0` and reached `main` through PR #9
(`f9806bb`), closing #2–#5;
offline 2494 passed, 35 xfailed. The unattended Workflow merged the
Responses ingress; Phase 12, the Responses backend, and Cohere stopped
under their rules and were finished interactively (Cohere redone by a
fresh Opus pass after its first pass failed spec review). Live spend: ledger
delta $0.5502, $0.55 of it unknown-cost placeholders. rdwj kept Cohere's
design calls as built. Record:
`session-summaries/2026-10-29-api-shapes-session-b.md`.

**Carried (no owner yet; pick up in "If blocked"):** Cohere review
should-fix items (tool results as bare strings, fixture test depth,
`fn["name"]` KeyError, 96-text embed limit); `openrouter_responses`
refusing `parallel_tool_calls` from one model's behavior; Responses live
test cost via `usage.cost`; `library/client.py` and
`tests/harness/scenarios.py` over the size guideline.

## What landed earlier (2026-10-29): Phase 8, Claude Code acceptance

Phase 8 done, `95e1f34..4415535`: Claude Code completes a tool-using task
through the gateway on Ollama, Claude via the anthropic provider, and
OpenRouter Chat Completions, for $0.0035. One gateway fix (Ollama thinking
on models that can't think). Closed #1. Record:
`session-summaries/2026-10-29-api-shapes-phase8.md`.

## What landed earlier (2026-10-29): Phase 11 slice 2, reconcile, 0.2.0 released

`model-calling==0.2.0` is on PyPI (tag `v0.2.0` at `f88dcc8`); the rest of
the epic is GitHub issues #1–#8 (milestone 0.3.0). Record:
`session-summaries/2026-10-29-api-shapes-phase11-slice2.md`.

## What landed earlier (2026-10-29): Phase 11 slice 2, Phase 11 complete

All five items, `c6e001b..53ca328`: `/v1/models/refresh` and the legacy
registry gone (moderations kept), `/v1/config/update` and the admin token
gone, `Client(models=...)` with alias-wins, examples on the library,
compose mounting `examples/config/`, and a hand-run swap with no restart.
Also fixed Windows CI. Record:
`session-summaries/2026-10-29-api-shapes-phase11-slice2.md`.

## What landed earlier (2026-10-29): Phase 11 slice 1

The gateway serves models by alias from a hot-reloaded YAML file,
`10df304..ddbf44b`: no backend name crosses the wire (responses, errors,
and `/v1/models` show aliases; every other name is 404), embedding
`dimensions` are enforced, and an edit switches backends with no restart.
Review fixes: error-message scrubbing, reload robustness, live tests on
aliases (not run). Record:
`session-summaries/2026-10-29-api-shapes-phase11-slice1.md`.

## What landed earlier (2026-10-29): Phase 10 complete

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

Still open from earlier slices: owned by Phase 12 since the 2026-10-29
reconciliation.

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

- **`PYPI_README.md` is the published readme and has drifted** from
  `README.md` (it predates the Anthropic backend). Sync it in Phase 9
  before 0.2.0.
- **The live suite has never run on aliases** (moved in Phase 11 slice 1).
  Phase 8's live runs will exercise the gateway path; run `-m live` once
  if anything there looks off.

## If blocked

- PyPI or the release workflow fails: fix forward in model-calling
  (that's the session), or pilot sanicode against
  `model-calling @ git+https://github.com/rdwj/model-calling@<sha>` and
  repin once 0.3.0 is out.
- sanicode blocked: take the Session B carries above, one commit each,
  offline, starting with Cohere's tool-result shape and the Responses live
  test's `usage.cost`.
- Then #6's offline item: the "Migrating from LiteLLM" doc (Phase 9
  slice 1 item 1).
