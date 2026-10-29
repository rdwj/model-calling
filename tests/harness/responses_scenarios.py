"""
Each grid scenario in OpenAI Responses form, where Responses can express it.

`RESPONSES_FORMS` maps a scenario to the Responses fields beyond `model`,
`max_output_tokens`, and `stream`, which come from the scenario, as does
`input` when a form leaves it out (the scenario's prompt). A
scenario missing here has no Responses form: Responses has no `stop` or
`seed` (system_blocks, max_completion_tokens), no thinking budget
(thinking), and always reports usage (stream_usage is stream's request).

`REJECTIONS` are the requests the stateless ingress refuses before any
backend call, each with the name its 400 must mention.
"""

import json

from tests.harness.scenarios import (
    CAPITAL_SCHEMA,
    IMAGE_B64,
    IMAGE_PROMPT,
    PRIOR_CALL_ID,
    PRIOR_CALL_INPUT,
    STRICT_WEATHER_TOOL,
    TEXT_PROMPT,
    TIME_TOOL,
    TOOL_ERROR_TEXT,
    TOOL_PROMPT,
    WEATHER_TOOL,
    Scenario,
)


def _function_tool(chat_tool: dict) -> dict:
    """A Chat Completions tool as a Responses function tool. `strict` is
    required; null leaves it to the backend, as Chat Completions' absent one does."""
    fn = chat_tool["function"]
    tool = {"type": "function", "name": fn["name"], "description": fn["description"]}
    return {**tool, "parameters": fn.get("parameters"), "strict": fn.get("strict")}


WEATHER = _function_tool(WEATHER_TOOL)
TIME = _function_tool(TIME_TOOL)
STRICT_WEATHER = _function_tool(STRICT_WEATHER_TOOL)

_TOOLS = {"tools": [WEATHER]}

RESPONSES_FORMS = {
    "text": {},
    "tools": _TOOLS,
    "stream": {},
    "stream_tools": _TOOLS,
    "backend_error": {},
    "tool_error": {
        "input": [
            {"role": "user", "content": TOOL_PROMPT},
            {
                "type": "function_call",
                "call_id": PRIOR_CALL_ID,
                "name": "get_weather",
                "arguments": json.dumps(PRIOR_CALL_INPUT),
            },
            {"type": "function_call_output", "call_id": PRIOR_CALL_ID, "output": TOOL_ERROR_TEXT},
        ],
        "tools": [WEATHER],
    },
    "parallel_tools": _TOOLS,
    "image": {
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{IMAGE_B64}",
                        "detail": "auto",
                    },
                    {"type": "input_text", "text": IMAGE_PROMPT},
                ],
            }
        ]
    },
    "thinking_unsupported": {"reasoning": {"effort": "high"}},
    "json_schema": {
        "text": {"format": {"type": "json_schema", "name": "response", "schema": CAPITAL_SCHEMA}}
    },
    "reasoning_effort": {"reasoning": {"effort": "low"}},
    "tool_required": {**_TOOLS, "tool_choice": "required"},
    "tool_named": {
        "tools": [WEATHER, TIME],
        "tool_choice": {"type": "function", "name": "get_weather"},
    },
    "parallel_tools_off": {**_TOOLS, "parallel_tool_calls": False},
    "tool_strict": {"tools": [STRICT_WEATHER]},
    "tool_parameterless": {"tools": [WEATHER, {**TIME, "parameters": None}]},
    # The list form of input, with store asked for.
    "store": {"input": [{"role": "user", "content": TEXT_PROMPT}], "store": True},
}

# Scenario -> (the field that makes the stateless ingress refuse it, the
# name its 400 message must mention).
REJECTIONS = {
    "previous_response_id": ({"previous_response_id": "resp_earlier"}, "previous_response_id"),
    "conversation": ({"conversation": "conv_earlier"}, "conversation"),
    "background": ({"background": True}, "background"),
    "web_search": ({"tools": [{"type": "web_search"}]}, "web_search"),
    "file_search": (
        {"tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}]},
        "file_search",
    ),
    "code_interpreter": (
        {"tools": [{"type": "code_interpreter", "container": {"type": "auto"}}]},
        "code_interpreter",
    ),
    "computer_use": ({"tools": [{"type": "computer"}]}, "computer use"),
    "encrypted_reasoning": ({"include": ["reasoning.encrypted_content"]}, "encrypted_content"),
}
RESPONSES_FORMS.update({name: extra for name, (extra, _) in REJECTIONS.items()})


def responses_body(scenario: Scenario, model: str) -> dict:
    if not scenario.responses_form:
        raise ValueError(f"scenario {scenario.name!r} has no OpenAI Responses form")
    return {
        "model": model,
        "input": scenario.prompt,
        **RESPONSES_FORMS[scenario.name],
        "max_output_tokens": scenario.max_tokens,
        "stream": scenario.stream,
    }


def rejected_name(scenario_name: str) -> str:
    """What the 400 for a rejected scenario must name."""
    return REJECTIONS[scenario_name][1]
