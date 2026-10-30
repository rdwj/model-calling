"""
The headers a call puts on the wire: the `openai` prefix's default
`OpenAI-Organization` header goes only to its configured base URL, and
every call method's `extra_headers` go over everything else.
"""

import inspect
import json

import httpx
import pytest
import yaml

import model_calling.library as library_module
from model_calling.library import Client
from model_calling.providers import ORGANIZATION_HEADER

ORG = "org-from-env"
OPENAI_BASE = "https://api.openai.com/v1"
ELSEWHERE = "https://api.elsewhere.test/openai/v1"
GROQ = "https://api.groq.com/openai/v1"
USER = [{"role": "user", "content": "x"}]

COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
CHUNK = {
    "id": "chatcmpl-1",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "m",
    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}],
}
FINAL_CHUNK = {**CHUNK, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
SSE = "".join(f"data: {json.dumps(c)}\n\n" for c in (CHUNK, FINAL_CHUNK)) + "data: [DONE]\n\n"
EMBEDDING = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "model": "m",
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}


class Backend:
    """An OpenAI-compatible backend that records every request it gets."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json=EMBEDDING)
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, text=SSE, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=COMPLETION)

    @property
    def headers(self) -> httpx.Headers:
        assert len(self.requests) == 1, [str(r.url) for r in self.requests]
        return self.requests[0].headers


@pytest.fixture
def backend(monkeypatch) -> Backend:
    monkeypatch.setenv("OPENAI_ORG_ID", ORG)
    return Backend()


def client_for(backend: Backend, **openai) -> Client:
    return Client(config={"openai": {"api_key": "k", "transport": backend.transport, **openai}})


# Every call method, as (method name, positional arguments, keyword arguments).
CALLS = {
    "completion": ("completion", ("openai/m", USER), {}),
    "completion_stream": ("completion", ("openai/m", USER), {"stream": True}),
    "messages": ("messages", ("openai/m", USER), {"max_tokens": 8}),
    "messages_stream": ("messages", ("openai/m", USER), {"max_tokens": 8, "stream": True}),
    "responses": ("responses", ("openai/m", "x"), {}),
    "responses_stream": ("responses", ("openai/m", "x"), {"stream": True}),
    "embedding": ("embedding", ("openai/m", "x"), {}),
}


async def call(target, name: str, is_async: bool, args, kwargs):
    """Make the call `name` (or its async form) on `target` and read it to the end."""
    method = getattr(target, f"a{name}" if is_async else name)
    result = method(*args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    if hasattr(result, "__aiter__"):
        return [item async for item in result]
    if hasattr(result, "__next__"):
        return list(result)
    return result


async def simple_call(client: Client, **kwargs):
    return await call(client, "completion", True, ("openai/m", USER), kwargs)


@pytest.mark.parametrize(
    "api_base, sends_org",
    [(None, True), (OPENAI_BASE, True), (OPENAI_BASE + "/", True), (ELSEWHERE, False)],
    ids=["default", "same_base", "same_base_slash", "elsewhere"],
)
async def test_organization_goes_only_to_the_configured_base(backend, api_base, sends_org):
    async with client_for(backend) as client:
        await simple_call(client, api_base=api_base)
    assert backend.headers.get(ORGANIZATION_HEADER) == (ORG if sends_org else None)


async def test_organization_follows_a_configured_base_url(backend):
    """The rule is the prefix's configured base, whatever it is, not OpenAI's URL."""
    async with client_for(backend, base_url=ELSEWHERE) as client:
        await simple_call(client)
    assert backend.headers[ORGANIZATION_HEADER] == ORG


@pytest.mark.parametrize(
    "configured, alias_base, sends_org",
    [
        (None, OPENAI_BASE, True),
        (ELSEWHERE, ELSEWHERE, True),
        (None, GROQ, False),
        (ELSEWHERE, GROQ, False),
    ],
    ids=["default_base", "configured_base", "elsewhere", "elsewhere_from_configured"],
)
async def test_alias_base_url_counts_as_a_per_call_api_base(
    backend, tmp_path, configured, alias_base, sends_org
):
    """An alias target's `base_url` reaches the call as its `api_base`, so the
    organization goes only where the prefix's configured base would send it."""
    path = tmp_path / "models.yaml"
    document = {
        "backends": {"remote": {"provider": "openai", "base_url": alias_base}},
        "models": {"chat": {"backend": "remote", "model": "m"}},
    }
    path.write_text(yaml.safe_dump(document))
    openai = {"api_key": "k", "transport": backend.transport}
    if configured is not None:
        openai["base_url"] = configured
    async with Client(models=path, config={"openai": openai}) as client:
        await call(client, "completion", True, ("chat", USER), {})
    assert str(backend.requests[0].url).startswith(alias_base)
    assert backend.headers.get(ORGANIZATION_HEADER) == (ORG if sends_org else None)


async def test_configured_headers_still_go_to_any_api_base(backend):
    headers = {ORGANIZATION_HEADER: "org-configured", "X-Team": "a"}
    async with client_for(backend, headers=headers) as client:
        await simple_call(client, api_base=ELSEWHERE)
    assert backend.headers[ORGANIZATION_HEADER] == "org-configured"
    assert backend.headers["x-team"] == "a"


async def test_configured_headers_override_the_organization(backend):
    async with client_for(backend, headers={"openai-organization": "org-configured"}) as client:
        await simple_call(client)
    assert backend.headers.get_list(ORGANIZATION_HEADER) == ["org-configured"]


async def test_empty_organization_sends_none(backend):
    async with client_for(backend, organization="") as client:
        await simple_call(client)
    assert ORGANIZATION_HEADER not in backend.headers


# (extra_headers, api_base, header name, values on the wire)
EXTRA_HEADERS = {
    "adds": ({"X-Trace": "t1"}, None, "x-trace", ["t1"]),
    "overrides_auth": ({"Authorization": "Bearer c"}, None, "authorization", ["Bearer c"]),
    "overrides_org": ({ORGANIZATION_HEADER: "org-c"}, None, ORGANIZATION_HEADER, ["org-c"]),
    "overrides_config": ({"x-team": "b"}, None, "x-team", ["b"]),
    "org_elsewhere": ({ORGANIZATION_HEADER: "org-c"}, ELSEWHERE, ORGANIZATION_HEADER, ["org-c"]),
}


@pytest.mark.parametrize(
    "extra, api_base, name, expected", EXTRA_HEADERS.values(), ids=EXTRA_HEADERS
)
async def test_extra_headers_go_over_everything(backend, extra, api_base, name, expected):
    async with client_for(backend, headers={"X-Team": "a"}) as client:
        await simple_call(client, extra_headers=extra, api_base=api_base)
    assert backend.headers.get_list(name) == expected


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("name, args, kwargs", CALLS.values(), ids=CALLS)
@pytest.mark.parametrize("via", ["client", "module"])
async def test_every_call_method_sends_extra_headers(
    backend, monkeypatch, via, name, args, kwargs, is_async
):
    client = client_for(backend)
    monkeypatch.setattr(library_module, "_default", client)
    target = client if via == "client" else library_module
    extra = {"X-Trace": "t1", "authorization": "Bearer per-call"}
    try:
        await call(target, name, is_async, args, {**kwargs, "extra_headers": extra})
    finally:
        await client.aclose()
    assert backend.headers["x-trace"] == "t1"
    assert backend.headers.get_list("authorization") == ["Bearer per-call"]
    assert backend.headers[ORGANIZATION_HEADER] == ORG
