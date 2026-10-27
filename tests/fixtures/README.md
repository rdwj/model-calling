# Recorded backend fixtures

Each `<backend>/<scenario>.json` file is one exchange with a real backend,
recorded by `tests/harness/record.py` and replayed offline by
`tests/harness/replay.py`. The `replay.py` docstring describes the format.

The recorder sends each scenario's native request directly to the backend
instead of going through our adapters, so fixtures show what the backend
really returns even where our translation is broken.

| Backend | Source | Model |
| --- | --- | --- |
| `ollama` | local Ollama, `/api/chat` | `qwen3:4b` |
| `openai_compat` | OpenRouter, `/api/v1/chat/completions` | `mistralai/mistral-nemo` |

## Scrubbing

Request headers are never written. OpenRouter's `user_id` in error bodies
is replaced with `scrubbed`. The recorder refuses to save a fixture that
contains the API key, and `tests/test_harness.py` checks every committed
fixture for credential patterns.

## OpenRouter divergences

Places where OpenRouter's responses don't match the official SDK types:

- **Responses API usage** (`/api/v1/responses`, checked 2026-10-26):
  `usage.input_tokens_details` omits `cache_write_tokens`, which
  `openai.types.responses.ResponseUsage` requires as of `openai` 3.35.1.
  The live smoke test fills that one field before validating.
