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
transport answers each request with the first unused fixture whose method,
URL (without query), and JSON body match, and fails loudly on anything
else. A body mismatch names the top-level fields that differ. Nothing is
ignored: when what we send changes, re-record the fixture.
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

    def body_mismatch(self, request: httpx.Request) -> Optional[str]:
        """Describe how the sent body differs from the recorded one, or None.

        Compared as canonical JSON, so `true` never matches `1`.
        """
        try:
            sent = json.loads(request.content) if request.content else None
        except ValueError:
            return f"sent a body that isn't JSON: {request.content[:200]!r}"
        recorded = self.request_json
        if _canonical(sent) == _canonical(recorded):
            return None
        if not (isinstance(sent, dict) and isinstance(recorded, dict)):
            return f"recorded {recorded!r}, sent {sent!r}"
        fields = sorted(
            k
            for k in recorded.keys() | sent.keys()
            if _canonical(recorded.get(k, _ABSENT)) != _canonical(sent.get(k, _ABSENT))
        )
        return "; ".join(
            f"{k}: recorded {recorded.get(k, _ABSENT)!r}, sent {sent.get(k, _ABSENT)!r}"
            for k in fields
        )

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


_ABSENT = "<absent>"


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True)


def _strip_query(url: httpx.URL) -> str:
    return str(url.copy_with(query=None))


def fixture_path(backend: str, scenario: str) -> Path:
    return FIXTURES_DIR / backend / f"{scenario}.json"


class ReplayTransport(httpx.AsyncBaseTransport, httpx.BaseTransport):
    """Answer requests from recorded exchanges and keep what was sent."""

    def __init__(self, exchanges: List[Exchange]):
        self._pending = list(exchanges)
        self.requests: List[httpx.Request] = []
        # Every mismatch raised, kept because the code under test may catch
        # the exception and turn it into an ordinary error response.
        self.failures: List[str] = []

    def check(self) -> None:
        """Fail if any request went unanswered, even if the caller swallowed
        it, or if a recorded exchange was never requested."""
        if self.failures:
            raise AssertionError("Replay failed: " + " || ".join(self.failures))
        if self._pending:
            unused = "; ".join(f"{e.method} {e.url} ({e.source})" for e in self._pending)
            raise AssertionError(f"Recorded exchanges were never requested: {unused}")

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        mismatches = []
        for i, exchange in enumerate(self._pending):
            if not exchange.matches(request):
                continue
            mismatch = exchange.body_mismatch(request)
            if mismatch is None:
                del self._pending[i]
                return exchange.to_response()
            mismatches.append(f"{exchange.source}: {mismatch}")
        if mismatches:
            failure = (
                f"Request body for {request.method} {request.url} differs from the "
                f"recording. {' | '.join(mismatches)}"
            )
        else:
            available = [f"{e.method} {e.url} ({e.source})" for e in self._pending] or ["none"]
            failure = (
                f"No recorded exchange for {request.method} {request.url}. "
                f"Unused fixtures: {'; '.join(available)}"
            )
        self.failures.append(failure)
        raise AssertionError(failure)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.read()
        return self._answer(request)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        return self._answer(request)

    def sent_json(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)
