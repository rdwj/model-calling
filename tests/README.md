# Tests

There are three suites, selected by pytest marker.

| Suite | Command | Needs |
| --- | --- | --- |
| Offline (default) | `uv run pytest` | nothing outside the repo |
| Integration | `uv run pytest -m integration` | a running gateway on `MODEL_CALLING_URL` (default `http://localhost:8000`) backed by Ollama |
| Live | `uv run pytest -m live` | `OPENROUTER_API_KEY` in the environment; costs money |

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
  `httpx.ASGITransport` with a fresh registry. `gateway.use(backend,
  scenario)` points it at a backend and replays that fixture by setting
  the replay transport as the prefix config's `transport`, which the
  provider's pooled HTTP client picks up.
- `library.py` provides the `library` fixture, the same idea for the
  library API: `library.use(backend, scenario)` builds a fresh `Client`
  whose prefix config carries the replay transport.
- `grid.py` parametrizes tests over (ingress, backend, scenario) cells.
  The gateway's ingresses (`chat_completions`, `gateway_messages`,
  `gateway_embeddings`) run in `test_gateway_grid.py`, which also checks
  that every gateway answer equals the library's for the same cell. The
  library's `library_completion`, `library_messages`, and
  `library_embedding` run in `test_library_grid.py`. Embeddings have their
  own backends and scenarios (`EMBEDDING_BACKENDS`, `embedding_grid`).
  `KNOWN_FAILURES` lists the cells that are expected to fail as strict
  xfails, each naming the epic phase that fixes it (or the backend
  limitation behind it). When a fix lands, the run fails until the entry
  is removed. Chat Completions ingresses skip scenarios with no Chat
  Completions form, and a backend that `lacks` a capability a scenario
  `needs` (vLLM's gpt-oss has no vision) makes that cell a strict xfail.
- `scenarios.py` defines the requests and backends shared by the tests and
  the recorder. A scenario is written in both wire forms where both can
  express it; `native_form` picks the one the backend body is built from,
  so strict replay proves the other form sends the same body. A backend's
  `capability_models` names the model it uses for a scenario that needs
  vision or reasoning.
- `checks.py` holds what each scenario must return, checked on the typed
  result, shared by the library, gateway, and SDK-client cells.
- `sdk.py` validates output against the official `openai` and `anthropic`
  SDK types.
- `replay.py` serves recorded exchanges from `fixtures/`. A request must
  match a fixture's method, URL, and JSON body exactly; when what we send
  changes, re-record rather than loosening the match. Because the gateway
  turns a transport error into an HTTP error, the `gateway` fixture also
  fails the test at teardown if any replay rejected a request.

To add a scenario, define it in `scenarios.py`, record it (below), and add
its checks to `checks.py` (`COMPLETION_CHECKS` and `MESSAGE_CHECKS`). A backend with `in_grid=False` (`vllm_messages`) is
recorded and validated on its own until a provider can reach it.

## Fixtures and the recorder

`fixtures/<backend>/<scenario>.json` holds one recorded backend exchange.
See [fixtures/README.md](fixtures/README.md) for the format and for known
provider divergences. To record missing fixtures:

```bash
uv run python -m tests.harness.record
uv run python -m tests.harness.record --backend ollama --scenario tools --force
```

Ollama must be running with `qwen3:4b`, `gemma3`, and `nomic-embed-text` pulled, and recording OpenRouter
fixtures needs `OPENROUTER_API_KEY`. Recording vLLM needs `VLLM_BASE_URL`
(the server root, without `/v1`) and `VLLM_MODEL`, plus `VLLM_API_KEY` if
the endpoint wants one; `VLLM_VERIFY_TLS=false` skips certificate checks
for an endpoint reached through a port-forward. Native request bodies come
from each backend's provider, so recordings match what the gateway sends.
Run gitleaks on `tests/fixtures/` before committing new recordings.

## Live tests and the spend ledger

Live tests and the recorder append each paid OpenRouter call's cost to
`live/spend.jsonl`. Commit the ledger after every live run. Both refuse to
make calls once the ledger total reaches the $5 epic budget.

## Integration suite

`integration/` holds the older tests that call a running gateway over
HTTP. It is only collected when the run selects `-m integration`. The
`run_tests.sh` and `test_ollama.sh` scripts at the repo root run it, and
`auto_config_tests.sh` writes `tests/models_available.py` to choose its
Ollama models.
