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
| `vllm` | vLLM 0.24 on OpenShift AI, `/v1/chat/completions` | `gpt-oss-20b` |

The `vllm` recordings go to `VLLM_BASE_URL` and `VLLM_MODEL` from the
environment. The fixtures store the replay URL (`http://localhost:8000`) and
the model name `gpt-oss-20b` in their place, so deployment details stay out
of the repo. The served model needs `--enable-auto-tool-choice
--tool-call-parser openai` for the `tools` scenario; without the parser,
gpt-oss leaves its call in the reasoning text.

## Scrubbing

Request headers are never written. OpenRouter's `user_id` in error bodies
is replaced with `scrubbed`. The recorder refuses to save a fixture that
contains the API key, and `tests/test_harness.py` checks every committed
fixture for credential patterns.

## OpenRouter divergences

Places where OpenRouter's responses don't match the official SDK types:

- **Chat Completions streaming** (checked 2026-10-26): the final chunk
  that carries `usage` repeats `finish_reason` and a delta in `choices`,
  where OpenAI sends `choices: []`. The grid's stream check allows a
  repeated `finish_reason`.
- **Responses API usage** (`/api/v1/responses`, checked 2026-10-26):
  `usage.input_tokens_details` omits `cache_write_tokens`, which
  `openai.types.responses.ResponseUsage` requires as of `openai` 3.35.1.
  The live smoke test fills that one field before validating.

## vLLM divergences

Places where vLLM's Chat Completions responses (0.24, checked 2026-10-28)
don't match the official SDK types. The SDK models ignore unknown fields,
so all of these still validate:

- **Reasoning** arrives as `reasoning` on the message and on stream deltas.
  Older vLLM releases and some other servers use `reasoning_content`.
  Neither field is in `openai.types.chat`.
- **Extra fields**: choices carry `stop_reason` (an integer token id when a
  tool call ends the turn), `token_ids`, and `routed_experts`; the body
  carries `prompt_token_ids`, `prompt_text`, and `kv_transfer_params`.
- **Errors**: a missing model is HTTP 404 with an integer `code`
  (`{"error": {"code": 404, "type": "NotFoundError", ...}}`).
- **Streaming** sends no usage chunk unless the request sets
  `stream_options.include_usage`.
