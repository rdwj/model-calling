"""
Claude Code completes a tool-using task through the gateway's Messages
ingress, once per backend. See README.md for prerequisites.

    uv run pytest -m acceptance -k ollama

Each run starts the gateway in its own process with a fresh alias file,
points `claude -p` at it from an empty temp directory, and checks the files
the task leaves behind. Claude Code names models only by alias (`main` and
`small`), so no Claude id reaches the gateway. A paid run appends its cost
to tests/live/spend.jsonl.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import yaml

from tests.acceptance.gateway import COSTS_ENV
from tests.harness import ledger
from tests.harness.scenarios import OPENROUTER_URL
from tests.live.conftest import OPENROUTER_KEY_ENV

pytestmark = pytest.mark.acceptance

REPO_ROOT = Path(__file__).parents[2]
LEDGER_SOURCE = "tests/acceptance"
CLAUDE = "claude"
# The aliases Claude Code is told to use: one for its main model, one for
# its background (Haiku-class) work. Both map to the backend under test.
MAIN_ALIAS = "main"
SMALL_ALIAS = "small"
OLLAMA_MODEL_ENV = "MODEL_CALLING_ACCEPTANCE_OLLAMA_MODEL"
OLLAMA_MODEL = "qwen3-coder:30b"
GATEWAY_START_S = 30
GATEWAY_STOP_S = 10
CLAUDE_TIMEOUT_S = 900
LOG_TAIL_LINES = 40

# Each step names its tool, matching ALLOWED_TOOLS, so a model doesn't try a
# shell heredoc, get denied, and give up on Bash altogether. `--bare` offers
# only Bash, Edit, and Read; Edit creates a file when old_string is empty,
# which models don't assume, so the task says so.
TASK = (
    "Use the Edit tool with an empty old_string to create a Python script hello.py "
    "that writes the text hello to a file named out.txt. Then use the Edit tool to "
    "change hello.py so it writes hello, gateway instead. Then use the Bash tool to "
    "run exactly this command: python3 hello.py"
)
EXPECTED_OUTPUT = "hello, gateway"
# Enough to do the task and nothing else; anything that would prompt is denied.
ALLOWED_TOOLS = "Edit,Read,Bash(python3 *)"


@dataclass(frozen=True)
class Target:
    """A backend for the aliases: an alias-file backend entry and model."""

    backend: dict
    model: str
    paid: bool = False


OPENROUTER_MESSAGES = {
    "provider": "anthropic",
    "base_url": OPENROUTER_URL,
    "api_key_env": OPENROUTER_KEY_ENV,
}
OPENROUTER_CHAT = {"provider": "openrouter", "api_key_env": OPENROUTER_KEY_ENV}

TARGETS = {
    "ollama": Target({"provider": "ollama"}, os.environ.get(OLLAMA_MODEL_ENV, OLLAMA_MODEL)),
    # The cheapest Claude on OpenRouter ($0.10/M in, 2026-10-29), over the
    # anthropic provider. It never thinks, which this task doesn't need.
    "anthropic": Target(OPENROUTER_MESSAGES, "anthropic/claude-haiku-5.5", paid=True),
    # A non-Claude model over Chat Completions ($0.15/M in, 2026-10-29).
    "openrouter": Target(OPENROUTER_CHAT, "openai/gpt-oss-120b", paid=True),
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    if not path.exists():
        return f"({path.name} was not written)"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def write_aliases(path: Path, target: Target) -> None:
    model = {"backend": "under_test", "model": target.model}
    config = {
        "backends": {"under_test": target.backend},
        "models": {MAIN_ALIAS: model, SMALL_ALIAS: model},
    }
    path.write_text(yaml.safe_dump(config))


def start_gateway(tmp: Path, aliases: Path, costs: Path, port: int) -> subprocess.Popen:
    """`python -m model_calling` with the cost log, from a directory with no
    `.env`, so only the environment and the alias file configure it."""
    env = {
        **os.environ,
        "MODEL_CALLING_CONFIG": str(aliases),
        "PORT": str(port),
        COSTS_ENV: str(costs),
        "PYTHONPATH": str(REPO_ROOT),
    }
    log = (tmp / "gateway.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.acceptance.gateway"],
        cwd=tmp,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + GATEWAY_START_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"the gateway exited at startup:\n{tail(tmp / 'gateway.log')}")
        try:
            httpx.get(f"http://127.0.0.1:{port}/", timeout=1).raise_for_status()
            return proc
        except httpx.HTTPError:
            time.sleep(0.25)
    proc.terminate()
    pytest.fail(f"the gateway didn't answer in {GATEWAY_START_S}s:\n{tail(tmp / 'gateway.log')}")


def stop_gateway(proc: subprocess.Popen) -> None:
    """Stop the gateway, killing it if it ignores SIGTERM, so a hang never
    hides the run's own failure."""
    proc.terminate()
    try:
        proc.wait(timeout=GATEWAY_STOP_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def claude_env(home: Path, port: int) -> dict:
    """Only what Claude Code needs, with its home and config in the temp
    directory, so no user settings, login, or memory reach the run."""
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "TERM": "dumb",
        "CLAUDE_CONFIG_DIR": str(home),
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        # The gateway takes no client key; Claude Code needs one to start.
        "ANTHROPIC_API_KEY": "acceptance-not-a-real-key",
        "ANTHROPIC_MODEL": MAIN_ALIAS,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": MAIN_ALIAS,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": MAIN_ALIAS,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": SMALL_ALIAS,
        "ANTHROPIC_SMALL_FAST_MODEL": SMALL_ALIAS,
        "CLAUDE_CODE_SUBAGENT_MODEL": MAIN_ALIAS,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
    }


def run_claude(work: Path, home: Path, port: int) -> subprocess.CompletedProcess:
    command = [
        CLAUDE,
        "--bare",
        "-p",
        "--model",
        MAIN_ALIAS,
        "--no-session-persistence",
        "--output-format",
        "json",
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        ALLOWED_TOOLS,
    ]
    return subprocess.run(
        command,
        input=TASK,
        cwd=work,
        env=claude_env(home, port),
        capture_output=True,
        text=True,
        timeout=CLAUDE_TIMEOUT_S,
    )


def record_spend(name: str, target: Target, costs: Path) -> float:
    """Sum the run's reported costs, look up the rest by generation id, and
    append one ledger entry. A call whose cost stays unknown counts as
    ledger.UNKNOWN_COST_USD."""
    lines = costs.read_text().splitlines() if costs.exists() else []
    entries = [json.loads(line) for line in lines if line.strip()]
    total = sum(e["cost"] for e in entries if "cost" in e)
    # Chat Completions streams carry no cost; OpenRouter's generation
    # lookup takes their `gen-` ids (and rejects Messages' `msg_` ones).
    ids = [e["id"] for e in entries if str(e.get("id")).startswith("gen-")]
    key = os.environ[OPENROUTER_KEY_ENV]
    with httpx.Client(timeout=60) as client:
        for response_id in ids:
            total += ledger.response_cost(client, key, {"id": response_id}) or (
                ledger.UNKNOWN_COST_USD
            )
    calls = sum(1 for e in entries if "id" in e)
    ledger.record(
        LEDGER_SOURCE,
        f"claude code /v1/messages ({name}, {calls} calls)",
        target.model,
        None,
        total,
    )
    return total


@pytest.mark.parametrize("name", TARGETS)
def test_claude_code_task(name, tmp_path):
    target = TARGETS[name]
    if shutil.which(CLAUDE) is None:
        pytest.fail("Claude Code (`claude`) must be on PATH; see tests/acceptance/README.md")
    if target.paid:
        if not os.environ.get(OPENROUTER_KEY_ENV):
            pytest.fail(f"{OPENROUTER_KEY_ENV} must be set for the {name} run")
        ledger.check_budget()

    aliases, costs = tmp_path / "model-calling.yaml", tmp_path / "costs.jsonl"
    work, home = tmp_path / "work", tmp_path / "home"
    work.mkdir()
    home.mkdir()
    write_aliases(aliases, target)
    port = free_port()
    gateway = start_gateway(tmp_path, aliases, costs, port)
    try:
        result = run_claude(work, home, port)
    finally:
        stop_gateway(gateway)
        if target.paid:
            record_spend(name, target, costs)

    context = (
        f"claude exit {result.returncode}\nstdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}\ngateway log:\n{tail(tmp_path / 'gateway.log')}"
    )
    assert result.returncode == 0, context
    outcome = json.loads(result.stdout)
    assert not outcome["is_error"], context
    script, output = work / "hello.py", work / "out.txt"
    assert script.exists() and EXPECTED_OUTPUT in script.read_text(), context
    assert output.exists(), f"hello.py never ran\n{context}"
    assert output.read_text().strip() == EXPECTED_OUTPUT, context
