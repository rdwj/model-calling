"""OpenAI Responses types for the public responses() API: the response and
the stream events the library emits."""

from typing import Annotated, Any, Literal

from pydantic import Field, Tag, TypeAdapter

from model_calling.types.base import UNKNOWN, WireModel, known_or_unknown

ItemStatus = Literal["in_progress", "completed", "incomplete"]


class InputTokensDetails(WireModel):
    cache_write_tokens: int
    cached_tokens: int


class OutputTokensDetails(WireModel):
    reasoning_tokens: int


class ResponseUsage(WireModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    output_tokens_details: OutputTokensDetails
    total_tokens: int


class ResponseOutputText(WireModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list  # Keep loose; we don't emit annotations
    logprobs: list | None = None


class ResponseOutputMessage(WireModel):
    """Text the model wrote."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    status: ItemStatus
    content: list[ResponseOutputText]


class ResponseFunctionToolCall(WireModel):
    """A call to one of the request's function tools."""

    id: str | None = None
    type: Literal["function_call"] = "function_call"
    status: ItemStatus | None = None
    arguments: str
    call_id: str
    name: str


class Summary(WireModel):
    type: Literal["summary_text"] = "summary_text"
    text: str


class ResponseReasoningItem(WireModel):
    """The model's thinking, carried as summary text."""

    id: str
    type: Literal["reasoning"] = "reasoning"
    status: ItemStatus | None = None
    summary: list[Summary]


class UnknownOutputItem(WireModel):
    """Fallback for output item types this module doesn't model."""

    type: str


OutputItem = Annotated[
    Annotated[ResponseOutputMessage, Tag("message")]
    | Annotated[ResponseFunctionToolCall, Tag("function_call")]
    | Annotated[ResponseReasoningItem, Tag("reasoning")]
    | Annotated[UnknownOutputItem, Tag(UNKNOWN)],
    known_or_unknown("message", "function_call", "reasoning"),
]


class Response(WireModel):
    """An OpenAI Responses response. Nothing is stored, so `store` is false."""

    id: str
    created_at: int | float
    error: dict | None = None
    incomplete_details: dict | None = None
    instructions: str | None = None
    model: str
    object: Literal["response"] = "response"
    output: list[OutputItem]
    parallel_tool_calls: bool
    temperature: float | None = None
    tool_choice: str | dict[str, Any]
    tools: list[dict[str, Any]]  # The request's function tools, echoed
    top_p: float | None = None
    max_output_tokens: int | None = None
    reasoning: dict | None = None
    status: Literal["completed", "incomplete", "in_progress"] | None = None
    text: dict | None = None
    usage: ResponseUsage | None = None
    store: bool = False

    @property
    def output_text(self) -> str:
        """The text of every message item, joined, as the SDK's property gives it."""
        return "".join(
            part.text
            for item in self.output
            if item.type == "message"
            for part in item.content
            if part.type == "output_text"
        )


# --- stream events ----------------------------------------------------------


class _Event(WireModel):
    sequence_number: int


class _ItemEvent(_Event):
    item_id: str
    output_index: int


class ResponseCreatedEvent(_Event):
    type: Literal["response.created"] = "response.created"
    response: Response


class ResponseInProgressEvent(_Event):
    type: Literal["response.in_progress"] = "response.in_progress"
    response: Response


class ResponseCompletedEvent(_Event):
    type: Literal["response.completed"] = "response.completed"
    response: Response


class ResponseIncompleteEvent(_Event):
    type: Literal["response.incomplete"] = "response.incomplete"
    response: Response


class ResponseOutputItemAddedEvent(_Event):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int
    item: OutputItem


class ResponseOutputItemDoneEvent(_Event):
    type: Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int
    item: OutputItem


class ResponseContentPartAddedEvent(_ItemEvent):
    type: Literal["response.content_part.added"] = "response.content_part.added"
    content_index: int
    part: ResponseOutputText


class ResponseContentPartDoneEvent(_ItemEvent):
    type: Literal["response.content_part.done"] = "response.content_part.done"
    content_index: int
    part: ResponseOutputText


class ResponseTextDeltaEvent(_ItemEvent):
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    content_index: int
    delta: str
    logprobs: list


class ResponseTextDoneEvent(_ItemEvent):
    type: Literal["response.output_text.done"] = "response.output_text.done"
    content_index: int
    text: str
    logprobs: list


class ResponseFunctionCallArgumentsDeltaEvent(_ItemEvent):
    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    delta: str


class ResponseFunctionCallArgumentsDoneEvent(_ItemEvent):
    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"
    arguments: str


class ResponseReasoningSummaryPartAddedEvent(_ItemEvent):
    type: Literal["response.reasoning_summary_part.added"] = "response.reasoning_summary_part.added"
    summary_index: int
    part: Summary


class ResponseReasoningSummaryPartDoneEvent(_ItemEvent):
    type: Literal["response.reasoning_summary_part.done"] = "response.reasoning_summary_part.done"
    summary_index: int
    part: Summary


class ResponseReasoningSummaryTextDeltaEvent(_ItemEvent):
    type: Literal["response.reasoning_summary_text.delta"] = "response.reasoning_summary_text.delta"
    summary_index: int
    delta: str


class ResponseReasoningSummaryTextDoneEvent(_ItemEvent):
    type: Literal["response.reasoning_summary_text.done"] = "response.reasoning_summary_text.done"
    summary_index: int
    text: str


class ResponseErrorEvent(_Event):
    """The backend failed mid-stream."""

    type: Literal["error"] = "error"
    code: str | None = None
    message: str
    param: str | None = None


ResponseStreamEvent = Annotated[
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseCompletedEvent
    | ResponseIncompleteEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
    | ResponseContentPartAddedEvent
    | ResponseContentPartDoneEvent
    | ResponseTextDeltaEvent
    | ResponseTextDoneEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
    | ResponseReasoningSummaryPartAddedEvent
    | ResponseReasoningSummaryPartDoneEvent
    | ResponseReasoningSummaryTextDeltaEvent
    | ResponseReasoningSummaryTextDoneEvent
    | ResponseErrorEvent,
    Field(discriminator="type"),
]

_event_adapter = TypeAdapter(ResponseStreamEvent)


def parse_stream_event(data: dict) -> ResponseStreamEvent:
    """Parse a Responses stream event dict into its typed model."""
    return _event_adapter.validate_python(data)
