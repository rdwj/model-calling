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
| `vllm_messages` | the same vLLM, `/v1/messages` | `gpt-oss-20b` |
| `ollama_embed` | local Ollama 0.40, `/api/embed` | `nomic-embed-text` |
| `openai_compat_embed` | local Ollama 0.40, `/v1/embeddings` | `nomic-embed-text` |

Scenarios that need a capability the default model lacks use another model
on the same backend: `image` uses `gemma3` on Ollama and
`google/gemma-3-4b-it` on OpenRouter, and `thinking` uses
`openai/gpt-oss-20b` on OpenRouter. vLLM's gpt-oss-20b takes no images, so
vLLM has no `image` fixture and its cells are strict xfails. `stream_usage`
replays each backend's `stream` fixture, since its native request is the
same.

The `vllm` recordings go to `VLLM_BASE_URL` and `VLLM_MODEL` from the
environment. The fixtures store the replay URL (`http://localhost:8000`) and
the model name `gpt-oss-20b` in their place, so deployment details stay out
of the repo. `vllm_messages` is recorded the same way. Its request bodies
come from `formats.anthropic`, since no provider speaks Messages to a
backend yet, and its fixtures stay off the grid until one does (Phase 6).
They are validated on their own in `tests/test_vllm_messages_fixtures.py`.
The served model needs `--enable-auto-tool-choice
--tool-call-parser openai` for the `tools` scenario; without the parser,
gpt-oss leaves its call in the reasoning text.

`openai_compat_embed` stands in for vLLM's `/v1/embeddings`. The vLLM
deployment serves a generative model, which has no embeddings endpoint, so
these fixtures come from Ollama's OpenAI-compatible root instead: the same
wire format, recorded for free. The grid reaches them through the `vllm/`
prefix with `base_url` set to Ollama's `/v1`, which exercises the
OpenAI-compatible provider shared by `vllm/`, `openai/`, and `openrouter/`.

## Scrubbing

Request headers are never written. OpenRouter's `user_id` in error bodies
is replaced with `scrubbed`. The recorder refuses to save a fixture that
contains the API key, and `tests/test_harness.py` checks every committed
fixture for credential patterns.

## OpenRouter divergences

Places where OpenRouter's responses don't match the official SDK types:

- **Chat Completions streaming** (checked 2026-10-26): the final chunk
  that carries `usage` repeats `finish_reason` and a delta in `choices`,
  where OpenAI sends `choices: []`. Our stream parser emits one stop
  reason per stream, and the grid checks that our output carries exactly one.
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

## vLLM Messages divergences

Places where vLLM's Anthropic Messages endpoint (0.24, checked 2026-10-28)
departs from the `anthropic` 2.0.0 types or from Anthropic's own API:

- **Thinking signatures** are 32 hex characters minted by vLLM, on both
  the non-streaming block and the stream's `signature_delta`. They
  validate, but Anthropic would reject them if a client sent them back.
- **Stream thinking start**: `content_block_start` for a thinking block is
  `{"type": "thinking", "thinking": ""}` with no `signature`, which
  `ThinkingBlock` requires. Anthropic sends `"signature": ""` there. The
  fixture test fills that one field before validating.
- **Errors**: a missing model is HTTP 404 with
  `{"type": "error", "error": {"type": "NotFoundError", ...}}`. The SDK
  expects `not_found_error`, so `ErrorResponse` rejects it (a strict xfail
  in the fixture test). There is no `request_id`.
- **Ids**: messages use `chatcmpl-` ids and tool calls `chatcmpl-tool-`
  ids, where Anthropic uses `msg_` and `toolu_`. Nothing validates the
  prefix.
- **Omitted fields**: non-streaming bodies leave out `stop_sequence` and
  the cache fields of `usage`; the SDK defaults them to null.
