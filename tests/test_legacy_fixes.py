"""
Regression tests for bugs in the legacy code that is still reachable: the
embedding adapters, the Anthropic and Cohere non-streaming path, and the
moderation endpoint.
"""

import subprocess
import sys

import pytest

from model_calling.adapter import create_openai_response

# Imports every model_calling module except the entry point (which may
# configure logging) and the torch-dependent moderation adapter, then fails
# naming the first one that configured the root logger.
_IMPORT_ALL = """
import importlib, logging, pkgutil, sys
import model_calling
skip = {"model_calling.__main__", "model_calling.granite_guardian_adapter"}
for m in pkgutil.walk_packages(model_calling.__path__, "model_calling."):
    if m.name in skip:
        continue
    importlib.import_module(m.name)
    if logging.root.handlers:
        sys.exit(f"importing {m.name} configured the root logger")
"""


def test_importing_the_library_leaves_root_logging_alone():
    # A fresh interpreter: pytest has already configured logging in this one.
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_ALL], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_legacy_response_ids_are_unique_within_a_second():
    ids = {create_openai_response(model="m", content="x")["id"] for _ in range(5)}
    assert len(ids) == 5, ids


@pytest.mark.parametrize(
    "body",
    [{"input": "hello"}, {"input": "hello", "model": "text-moderation-latest"}],
    ids=["default_model", "unprefixed_model"],
)
async def test_moderation_never_resolves_to_a_chat_backend(gateway, monkeypatch, body):
    """Unprefixed moderation models (including the granite-guardian default)
    used to resolve to the Ollama chat adapter. Moderation must go to Granite
    Guardian, here reported missing since its extra isn't installed."""
    monkeypatch.setitem(sys.modules, "model_calling.granite_guardian_adapter", None)
    resp = await gateway.client.post("/v1/moderations", json=body)
    assert resp.status_code == 501, resp.text
    assert "model-calling[moderation]" in resp.json()["detail"]
