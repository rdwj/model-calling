"""
Serve recorded backend responses through an httpx transport.

A fixture file holds one backend exchange:

    {
      "meta": {"backend": ..., "scenario": ..., "recorded_at": ...},
      "request": {"method": "POST", "url": "...", "json": {...}},
      "response": {"status": 200, "content_type": "...",
                   "json": {...}}          # non-streaming
                   or "lines": ["...", ...]  # streaming body, one entry per line
    }

Request headers are never stored, so fixtures carry no credentials. The
transport answers each request with the first unused fixture whose method
and URL (without query) match, and fails loudly on anything else.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import httpx

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@dataclass
class Exchange:
    method: str
    url: str
    request_json: Optional[dict]
    status: int
    content_type: str
    response_json: Optional[dict] = None
    lines: Optional[List[str]] = None
    meta: dict = field(default_factory=dict)
    source: str = "<memory>"

    @classmethod
    def load(cls, path: Path) -> "Exchange":
        data = json.loads(path.read_text(encoding="utf-8"))
        req, resp = data["request"], data["response"]
        return cls(
            method=req["method"],
            url=req["url"],
            request_json=req.get("json"),
            status=resp["status"],
            content_type=resp["content_type"],
            response_json=resp.get("json"),
            lines=resp.get("lines"),
            meta=data.get("meta", {}),
            source=str(path),
        )

    def to_dict(self) -> dict:
        response = {"status": self.status, "content_type": self.content_type}
        if self.lines is not None:
            response["lines"] = self.lines
        else:
            response["json"] = self.response_json
        return {
            "meta": self.meta,
            "request": {"method": self.method, "url": self.url, "json": self.request_json},
            "response": response,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def matches(self, request: httpx.Request) -> bool:
        return request.method == self.method and _strip_query(request.url) == self.url

    def body_bytes(self) -> bytes:
        if self.lines is not None:
            return ("\n".join(self.lines) + "\n").encode("utf-8")
        return json.dumps(self.response_json).encode("utf-8")

    def to_response(self) -> httpx.Response:
        return httpx.Response(
            self.status,
            headers={"content-type": self.content_type},
            content=self.body_bytes(),
        )


def _strip_query(url: httpx.URL) -> str:
    return str(url.copy_with(query=None))


def fixture_path(backend: str, scenario: str) -> Path:
    return FIXTURES_DIR / backend / f"{scenario}.json"


class ReplayTransport(httpx.AsyncBaseTransport, httpx.BaseTransport):
    """Answer requests from recorded exchanges and keep what was sent."""

    def __init__(self, exchanges: List[Exchange]):
        self._pending = list(exchanges)
        self.requests: List[httpx.Request] = []

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        for i, exchange in enumerate(self._pending):
            if exchange.matches(request):
                del self._pending[i]
                return exchange.to_response()
        available = [f"{e.method} {e.url} ({e.source})" for e in self._pending] or ["none"]
        raise AssertionError(
            f"No recorded exchange for {request.method} {request.url}. "
            f"Unused fixtures: {'; '.join(available)}"
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.read()
        return self._answer(request)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        return self._answer(request)

    def sent_json(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)
