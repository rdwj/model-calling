# Tests

There are two suites, selected by pytest marker.

| Suite | Command | Needs |
| --- | --- | --- |
| Offline (default) | `uv run pytest` | nothing outside the repo |
| Live | `uv run pytest -m live` | `OPENROUTER_API_KEY` in the environment; costs money |
| Acceptance | `uv run pytest -m acceptance` | Claude Code, Ollama, and `OPENROUTER_API_KEY`; see `acceptance/README.md` |

CI runs only the offline suite. To check that it doesn't lean on your
machine, run it with an empty home directory:

```bash
HOME=$(mktemp -d) uv run pytest
```

## Offline suite

`conftest.py` isolates every offline test. Home points at a scratch
directory, so `~/.model_calling/` is never read. `load_dotenv` is a no-op,
backend credentials are removed from the environment, the capability cache
lives in `tmp_path`, and httpx's network transports raise. A test that
needs a backend gets it from a recorded fixture.

The harness lives in `harness/`:

- `gateway.py` provides the `gateway` fixture: the FastAPI app behind
  `httpx.ASGITransport` with a fresh library client. `gateway.use(backend,
  scenario)` points it at a backend and replays that fixture by setting
  the replay transport as the prefix config's `transport`, which the
  provider's pooled HTTP client picks up.
- `library.py` provides the `library` fixture, the same idea for the
  library API: `library.use(backend, scenario)` builds a fresh `Client`
  whose prefix config carries the replay transport.
- `grid.py` parametrizes tests over (ingress, backend, scenario) cells.
  The gateway's ingresses (`chat_completions`, `gateway_messages`,
  `gateway_responses`, `gateway_embeddings`) run in
  `test_gateway_grid.py`, which also checks that every gateway answer
  equals the library's for the same cell. The library's
  `library_completion`, `library_messages`, `library_responses`, and
  `library_embedding` run in `test_library_grid.py`. In
  `test_sdk_grid.py`, `sdk_messages` drives the gateway with the official
  `anthropic.AsyncAnthropic` client and streams through its
  `messages.stream` helper, `sdk_completion` drives it with
  `openai.AsyncOpenAI` and streams through the SDK's own iterator, and
  `sdk_responses` calls `client.responses.create(...)` and runs each
  stream through the SDK's `ResponseStreamState`, which raises on an
  event out of order. Both SDKs run on httpx2, so they get an httpx2 ASGI
  transport, and the cells check the message, completion, or response
  the SDK assembles from a stream. Embeddings have their
  own backends and scenarios (`EMBEDDING_BACKENDS`, `embedding_grid`).
  `KNOWN_FAILURES` lists the cells that are expected to fail as strict
  xfails, each naming the epic phase that fixes it (or the backend
  limitation behind it). When a fix lands, the run fails until the entry
  is removed. Chat Completions ingresses skip scenarios with no Chat
  Completions form, Messages ingresses skip scenarios with no Messages
  form, and a backend that `lacks` a capability a scenario `needs`
  (vLLM's gpt-oss has no vision) makes that cell a strict xfail. A
  backend that `refuses` a scenario (Ollama can't force a tool call) has
  no fixture for it: the cell checks that strict mode refuses the request,
  naming the parameters, without calling the backend. The Responses
  ingresses, added after the others, get no cell at all where a backend
  limitation (`BACKEND_LIMITATIONS`, `lacks`) would fail it, and a
  `rejected` scenario (`previous_response_id`, a built-in tool) checks
  that the stateless ingress refuses it by name without calling the
  backend.
- `scenarios.py` defines the requests and backends shared by the tests and
  the recorder. A scenario is written in both wire forms where both can
  express it; `native_form` picks the one the backend body is built from,
  so strict replay proves the other form sends the same body. A backend
  that speaks Messages (`vllm_messages`) builds its body from the ingress's
  own form instead, and where the Chat Completions form's body differs it
  replays `<scenario>.chat_completions.json`. The Responses form of each
  scenario is in `responses_scenarios.py`; it replays whichever form's
  fixture has the same native body, so it needs no recordings of its own
  (`<scenario>.responses.json` if it ever does). A backend's
  `capability_models` names the model it uses for a scenario that needs
  vision or reasoning.
- `checks.py` holds what each scenario must return, checked on the typed
  result, shared by the library, gateway, and SDK-client cells;
  `responses_checks.py` does the same for the Responses ingresses.
- `sdk.py` validates output against the official `openai` and `anthropic`
  SDK types.
- `replay.py` serves recorded exchanges from `fixtures/`. A request must
  match a fixture's method, URL, and JSON body exactly; when what we send
  changes, re-record rather than loosening the match. Because the gateway
  turns a transport error into an HTTP error, the `gateway` fixture also
  fails the test at teardown if any replay rejected a request.

To add a scenario, define it in `scenarios.py`, record it (below), and add
its checks to `checks.py` (`COMPLETION_CHECKS` and `MESSAGE_CHECKS`). A
backend with `in_grid=False` is recorded and validated on its own until a
provider can reach it.

## Fixtures and the recorder

`fixtures/<backend>/<scenario>.json` holds one recorded backend exchange.
A body that depends on what the backend says about its model (Ollama's
thinking levels, from `/api/show`) also replays that lookup, recorded
alongside as `model_info-<model>.json`.
See [fixtures/README.md](fixtures/README.md) for the format and for known
provider divergences. To record missing fixtures:

```bash
uv run python -m tests.harness.record
uv run python -m tests.harness.record --backend ollama --scenario tools --force
```

Ollama must be running with `qwen3:4b`, `gemma3`, `gpt-oss:20b`, and `nomic-embed-text` pulled, and recording OpenRouter
fixtures needs `OPENROUTER_API_KEY`. Recording vLLM needs `VLLM_BASE_URL`
(the API root with `/v1`, as the gateway reads it) and `VLLM_MODEL`, plus `VLLM_API_KEY` if
the endpoint wants one; `VLLM_VERIFY_TLS=false` skips certificate checks
for an endpoint reached through a port-forward. Native request bodies come
from each backend's provider, so recordings match what the gateway sends.
Run gitleaks on `tests/fixtures/` before committing new recordings.

## Live tests and the spend ledger

Live tests and the recorder append each paid OpenRouter call's cost to
`live/spend.jsonl`. Commit the ledger after every live run. Both refuse to
make calls once the ledger total reaches the $5 epic budget.
