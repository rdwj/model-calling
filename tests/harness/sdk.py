"""
Validate our output against the official openai and anthropic SDK types.

Each helper parses with the SDK's pydantic model and returns the typed
object, so a test fails with the SDK's own validation error when our wire
format drifts.
"""

import json
from typing import Any, List, Union

from anthropic.types import Message, RawMessageStreamEvent
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response
from pydantic import TypeAdapter

SSE_DONE = "[DONE]"

_anthropic_event = TypeAdapter(RawMessageStreamEvent)


def validate_chat_completion(body: dict) -> ChatCompletion:
    return ChatCompletion.model_validate(body)


def validate_chat_completion_chunks(chunks: List[dict]) -> List[ChatCompletionChunk]:
    return [ChatCompletionChunk.model_validate(chunk) for chunk in chunks]


def validate_anthropic_message(body: dict) -> Message:
    return Message.model_validate(body)


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
        data = "\n".join(data_lines)
        if data == SSE_DONE:
            events.append(SSE_DONE)
            continue
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError as e:
            raise ValueError(f"SSE data is not JSON: {data!r}") from e
    return events
