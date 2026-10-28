"""
Validate our output against the official openai and anthropic SDK types.

Each helper parses with the SDK's pydantic model and returns the typed
object, so a test fails with the SDK's own validation error when our wire
format drifts.
"""

import json
from collections.abc import Iterator
from typing import Any, List, Union

from anthropic.types import ErrorResponse, Message, RawMessageStreamEvent
from anthropic.types.message_create_params import (
    MessageCreateParamsNonStreaming,
    MessageCreateParamsStreaming,
)
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)
from openai.types.responses import Response
from pydantic import TypeAdapter

SSE_DONE = "[DONE]"

_anthropic_event = TypeAdapter(RawMessageStreamEvent)
_anthropic_request = {
    False: TypeAdapter(MessageCreateParamsNonStreaming),
    True: TypeAdapter(MessageCreateParamsStreaming),
}
_chat_completion_request = {
    False: TypeAdapter(CompletionCreateParamsNonStreaming),
    True: TypeAdapter(CompletionCreateParamsStreaming),
}


def validate_chat_completion(body: dict) -> ChatCompletion:
    return ChatCompletion.model_validate(body)


def validate_chat_completion_chunks(chunks: List[dict]) -> List[ChatCompletionChunk]:
    return [ChatCompletionChunk.model_validate(chunk) for chunk in chunks]


def validate_embedding_response(body: dict) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse.model_validate(body)


def validate_anthropic_message(body: dict) -> Message:
    return Message.model_validate(body)


def validate_anthropic_error(body: dict) -> ErrorResponse:
    return ErrorResponse.model_validate(body)


def _materialize(value: Any) -> Any:
    """Consume every lazily validated iterator, forcing its validation."""
    if isinstance(value, (Iterator, list, tuple)):
        return [_materialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _materialize(v) for k, v in value.items()}
    return value


def validate_anthropic_request(body: dict) -> dict:
    """Validate a Messages request body against the SDK's request TypedDicts."""
    return _validate_request(_anthropic_request, body)


def validate_chat_completion_request(body: dict) -> dict:
    """Validate a Chat Completions request body against the SDK's request TypedDicts."""
    return _validate_request(_chat_completion_request, body)


def _validate_request(adapters: dict, body: dict) -> dict:
    """The SDKs type `Iterable` fields, which pydantic validates lazily, and
    their TypedDicts ignore unknown keys. So this consumes every iterator,
    then requires the validated result to equal the input, which rejects keys
    the SDK doesn't define and values pydantic had to coerce.
    """
    adapter = adapters[bool(body.get("stream"))]
    validated = _materialize(adapter.validate_python(body))
    if validated != body:
        raise ValueError(
            "request differs from its SDK-validated form (unknown or coerced fields):\n"
            f"{body!r}\nvalidated as:\n{validated!r}"
        )
    return validated


def validate_anthropic_stream(events: List[dict]) -> List[Any]:
    return [_anthropic_event.validate_python(event) for event in events]


def validate_response(body: dict) -> Response:
    return Response.model_validate(body)


def parse_sse(text: str) -> List[Union[dict, str]]:
    """Return each SSE event's `data` payload, JSON-decoded except `[DONE]`.

    Handles both OpenAI-style (`data:` only) and Anthropic-style (`event:`
    then `data:`) streams. Raises ValueError on a data line that isn't JSON,
    since a malformed event is exactly what these tests exist to catch.
    """
    events: List[Union[dict, str]] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        data_lines = [
            line[len("data:") :].lstrip() for line in block.split("\n") if line.startswith("data:")
        ]
        if not data_lines:
            continue
        data = "\n".join(data_lines).strip()
        if not data:
            continue
        if data == SSE_DONE:
            events.append(SSE_DONE)
            continue
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError as e:
            raise ValueError(f"SSE data is not JSON: {data!r}") from e
    return events
