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
| `openrouter_responses` | OpenRouter, `/api/v1/responses` | `mistralai/mistral-nemo` |
| `vllm` | vLLM 0.24 on OpenShift AI, `/v1/chat/completions` | `gpt-oss-20b` |
| `vllm_messages` | the same vLLM, `/v1/messages` | `gpt-oss-20b` |
| `ollama_embed` | local Ollama 0.40, `/api/embed` | `nomic-embed-text` |
| `openai_compat_embed` | local Ollama 0.40, `/v1/embeddings` | `nomic-embed-text` |
| `cohere` | hand-built from the Cohere v2 reference, `/v2/chat` | `command-a-03-2025` |
| `cohere_embed` | hand-built from the Cohere v2 reference, `/v2/embed` | `embed-v4.0` |

Scenarios that need a capability the default model lacks use another model
on the same backend: `image` uses `gemma3` on Ollama and
`google/gemma-3-4b-it` on OpenRouter, and `thinking` uses
`openai/gpt-oss-20b` on OpenRouter. vLLM's gpt-oss-20b takes no images, so
vLLM has no `image` fixture and its cells are strict xfails. `stream_usage`
replays each backend's `stream` fixture, since its native request is the
same.

The `vllm` recordings go to `VLLM_BASE_URL` and `VLLM_MODEL` from the
environment. The fixtures store the replay URL (`http://localhost:8000/v1`) and
the model name `gpt-oss-20b` in their place, so deployment details stay out
of the repo. `vllm_messages` is recorded the same way. Its request bodies
come from the Anthropic provider, which the grid reaches through the
`anthropic/` prefix pointed at vLLM. Scenarios whose Chat Completions form
gives a different Messages body (it can't carry `cache_control`,
`metadata`, or `is_error`) have a second fixture,
`<scenario>.chat_completions.json`. The fixtures are also validated against
the SDK types in `tests/test_vllm_messages_fixtures.py`.
The served model needs `--enable-auto-tool-choice
--tool-call-parser openai` for the `tools` scenario; without the parser,
gpt-oss leaves its call in the reasoning text.

`openai_compat_embed` stands in for vLLM's `/v1/embeddings`. The vLLM
deployment serves a generative model, which has no embeddings endpoint, so
these fixtures come from Ollama's OpenAI-compatible root instead: the same
wire format, recorded for free. The grid reaches them through the `vllm/`
prefix with `base_url` set to Ollama's `/v1`, which exercises the
OpenAI-compatible provider shared by `vllm/`, `openai/`, and `openrouter/`.

## Cohere v2 (hand-built)

There is no Cohere API key, so nothing under `cohere/` or `cohere_embed/`
was recorded, and Cohere has no live coverage. Each request body is what
the Cohere provider sends (strict replay holds it to that), checked by hand
against the reference; each response was written by hand from the
reference. `meta.hand_built_from` names the page each file follows.
`tests/test_cohere_fixtures.py` checks both halves against the reference's
field lists. The pages, read 2026-10-29:

- Chat: https://docs.cohere.com/reference/chat (request body, response,
  error bodies, and the `ChatMessageV2`, `ToolCallV2`, `Thinking`,
  `ResponseFormatV2`, and `Usage` types)
- Chat streaming: https://docs.cohere.com/reference/chat-stream (the
  `message-start` through `message-end` event types and their `delta`s)
- Embed: https://docs.cohere.com/reference/embed (request body, the
  `EmbedByTypeResponseEmbeddings` response, `ApiMeta`)
- Models: https://docs.cohere.com/docs/models and
  https://docs.cohere.com/docs/cohere-embed (names and embed-v4.0's
  dimensions)

What the fixtures follow from those pages:

- **Requests.** The system prompt is the first message, with role `system`
  (v2 has no top-level `system`); two system blocks stay two text blocks.
  Tool call `function.arguments` is a JSON string. `tool_choice` is
  `REQUIRED`; strict tools set the top-level `strict_tools`. Thinking is
  `{"type": "enabled", "token_budget": 1024}`. The JSON schema rides in
  `response_format.json_schema` with `type: json_object`. `top_p`, `top_k`,
  and `stop` would be `p`, `k`, and `stop_sequences`.
- **Responses.** `finish_reason` is upper case (`COMPLETE`, `TOOL_CALL`).
  Content is a list of `text` or `thinking` blocks (thinking text in the
  block's `thinking` field, with no signature). Tool turns carry the
  `tool_plan` string before `tool_calls`, as `command-a-03-2025` writes
  it; the provider turns the plan into a text block. Usage has both
  `billed_units` and `tokens`, with `tokens.input_tokens` larger by
  Cohere's prompt template, as in the reference's examples. Error bodies
  are `{"id", "message"}`, and a missing model is HTTP 404.
- **Streams.** Server-sent events, `event:` naming each payload's `type`.
  `message-start` has its `id` at the top level and `delta.message.role`;
  content events nest under `delta.message.content`; `tool-plan-delta`
  under `delta.message.tool_plan`; tool call events under
  `delta.message.tool_calls`; `message-end` carries `finish_reason` and
  `usage` in `delta`. Content blocks and tool calls count their `index`
  separately, from 0 each; the stream parser maps both into one sequence.
- **Embeddings.** `texts`, `input_type: search_document`, and
  `embedding_types: ["float"]`; the answer keys vectors by type
  (`embeddings.float`), at embed-v4.0's default 1536 dimensions, with
  `meta.billed_units.input_tokens` as the only count, as in the
  reference's examples.
- **Models.** `image` uses `command-a-vision-07-2025` and `thinking` uses
  `command-a-reasoning-08-2025`; `command-a-03-2025` takes neither images
  nor a thinking budget.

Three Cohere cells have no fixture, because v2 can't express their
request; they check the refusal instead (`refuses` in
`tests/harness/scenarios.py`). A named
`tool_choice` (`tool_named`): the reference allows only `REQUIRED` and
`NONE`. `parallel_tool_calls: false` (`parallel_tools_off`): there is no
such field. A thinking effort without a budget (`reasoning_effort`):
`Thinking` has only `type` and `token_budget`. Strict mode refuses each of
these; the gateway drops them.

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
- **Responses API parallel_tool_calls** (`/api/v1/responses`, checked 2026-10-29):
  `parallel_tool_calls: false` is accepted in the request but not honored;
  mistralai/mistral-nemo returns parallel tool calls anyway. The
  `openrouter_responses` backend refuses this scenario.

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
- **Forced tool calls** (`tool_choice` of type `tool`, `tool_named`) end
  with `stop_reason: "end_turn"` although the turn is a `tool_use` block;
  Anthropic reports `tool_use`. The Anthropic provider reports `tool_use`
  for any turn with a tool call that ends in `end_turn`, as the
  OpenAI-compatible provider does.
