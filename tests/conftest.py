"""
Shared pytest configuration.

The default run is offline. It needs no network, backend, or running
service, and reads nothing from the machine: not `~/.model_calling/`, not
`.env`, and not the generated `tests/models_available.py`. Tests marked
`integration` or `live` opt out of that isolation.
"""

import atexit
import ipaddress
import os
import re
import shutil
import socket
import tempfile
from pathlib import Path

import dotenv
import httpx
import httpx2
import pytest

pytest_plugins = ["tests.harness.gateway", "tests.harness.library"]

TESTS_DIR = Path(__file__).parent
INTEGRATION_DIR = TESTS_DIR / "integration"
OPT_OUT_MARKERS = ("integration", "live")

# Environment the package reads for backend URLs and credentials. Offline
# tests must never see the developer's real values.
BACKEND_ENV_VARS = (
    "OLLAMA_BASE_URL",
    "VLLM_BASE_URL",
    "VLLM_API_KEY",
    "MODEL_CALLING_ANTHROPIC_BASE_URL",
    "MODEL_CALLING_ADMIN_TOKEN",
    "VLLM_OPENAI_COMPATIBLE",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_ORG_ID",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_API_BASE",
    "COHERE_API_KEY",
    "COHERE_API_BASE",
    "OPENROUTER_API_KEY",
)

# This module loads before any test module imports model_calling. Point
# home at a temporary directory so offline tests don't read or write the
# real home. Integration tests talk to a service in another process, and
# live tests read only the environment, so neither needs the real home.
_SESSION_HOME = tempfile.mkdtemp(prefix="model-calling-tests-home-")
atexit.register(shutil.rmtree, _SESSION_HOME, ignore_errors=True)
os.environ["HOME"] = _SESSION_HOME
os.environ["USERPROFILE"] = _SESSION_HOME

# Only the gateway entry point calls load_dotenv() (tests/test_packaging.py
# checks that importing the server doesn't). Make it a no-op anyway so a
# developer's .env never configures the code under test. The integration conftest calls
# dotenv.main.load_dotenv directly when it wants the real thing.
dotenv.load_dotenv = lambda *args, **kwargs: False


def _selects_marker(config: pytest.Config, marker: str) -> bool:
    """True when the -m expression asks for `marker` rather than excluding it."""
    expr = config.getoption("markexpr") or ""
    return re.search(rf"(?<!not )\b{marker}\b", expr) is not None


def pytest_ignore_collect(collection_path: Path, config: pytest.Config):
    # Importing the integration package reads .env and models_available.py,
    # so skip it entirely unless the run selects integration tests.
    if collection_path == INTEGRATION_DIR and not _selects_marker(config, "integration"):
        return True
    return None


def _is_offline(item: pytest.Item) -> bool:
    return not any(item.get_closest_marker(m) for m in OPT_OUT_MARKERS)


@pytest.fixture(autouse=True)
def offline_isolation(request, monkeypatch, tmp_path):
    """Strip backend env vars and block the network."""
    if not _is_offline(request.node):
        yield
        return

    for var in BACKEND_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    real_connect = socket.socket.connect

    def guarded_connect(sock, address):
        # Loopback stays open: asyncio's self-pipe uses it on Windows.
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            host = address[0]
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = host == "localhost"
            if not loopback:
                raise RuntimeError(
                    f"Offline test tried to connect to {address!r}. Serve backend "
                    "responses from a fixture, or mark the test integration/live."
                )
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    # The package reaches backends only through httpx. Blocking its network
    # transports also catches loopback calls (a local Ollama or server) that
    # the socket guard lets through. Replay and ASGI transports are unaffected.
    def blocked_transport(transport, request):
        raise RuntimeError(
            f"Offline test sent {request.method} {request.url} over the network. "
            "Use the replay transport from tests/harness, or mark the test integration/live."
        )

    async def blocked_async_transport(transport, request):
        blocked_transport(transport, request)

    # anthropic 2.x has its own httpx fork, httpx2; the SDK grid uses it.
    for lib in (httpx, httpx2):
        monkeypatch.setattr(lib.HTTPTransport, "handle_request", blocked_transport)
        monkeypatch.setattr(lib.AsyncHTTPTransport, "handle_async_request", blocked_async_transport)
    yield
