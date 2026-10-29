"""
The library/gateway split: the core imports only httpx and pydantic, and the
gateway's entry point explains the `server` extra instead of failing with a
traceback. Each check runs in a fresh interpreter, since this one already
has the server modules loaded.
"""

import subprocess
import sys

import pytest

SERVER_MODULES = ("fastapi", "starlette", "uvicorn", "dotenv")

_IMPORT_CORE = f"""
import sys
import model_calling
loaded = {{m.split(".")[0] for m in sys.modules}}
server = sorted(loaded & set({SERVER_MODULES!r}))
assert not server, f"import model_calling loaded {{server}}"
"""

# Hides one server module, as if the extra weren't installed, then runs main().
_RUN_WITHOUT = """
import sys
sys.modules[{name!r}] = None
from model_calling.__main__ import main
main()
"""


def run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)


def test_importing_the_library_loads_no_server_modules():
    result = run(_IMPORT_CORE)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["uvicorn", "fastapi", "dotenv"])
def test_gateway_without_the_server_extra_prints_an_install_hint(name):
    result = run(_RUN_WITHOUT.format(name=name))
    assert result.returncode == 1
    assert "pip install 'model-calling[server]'" in result.stderr
    assert f"({name} is not installed)" in result.stderr
    assert "Traceback" not in result.stderr


DOTENV_VAR = "MODEL_CALLING_TEST_DOTENV_MARKER"

# Runs main() with uvicorn.run replaced by printing `expr`.
_RUN_MAIN = """
import os
import uvicorn
import model_calling
uvicorn.run = lambda *args, **kwargs: print({expr})
from model_calling.__main__ import main
main()
"""


def run_in(directory, code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=directory
    )


@pytest.fixture
def dotenv_dir(tmp_path):
    (tmp_path / ".env").write_text(
        f"{DOTENV_VAR}=from-dotenv\nOLLAMA_BASE_URL=http://ollama.from-dotenv:11434\n"
    )
    return tmp_path


def test_importing_the_server_does_not_read_dotenv(dotenv_dir):
    code = f"""
import os
before = dict(os.environ)
import model_calling.server
assert dict(os.environ) == before, sorted(set(os.environ) ^ set(before))
"""
    result = run_in(dotenv_dir, code)
    assert result.returncode == 0, result.stderr


def test_the_entry_point_reads_dotenv_from_the_working_directory(dotenv_dir):
    result = run_in(dotenv_dir, _RUN_MAIN.format(expr=f"os.environ.get({DOTENV_VAR!r})"))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "from-dotenv", result.stdout


def test_backend_urls_in_dotenv_reach_the_server(dotenv_dir):
    expr = "model_calling.server.registry.client.config['ollama']['base_url']"
    result = run_in(dotenv_dir, _RUN_MAIN.format(expr=expr))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "http://ollama.from-dotenv:11434"
    assert "ollama base URL: http://ollama.from-dotenv:11434" in result.stderr


def test_the_entry_point_ignores_dotenv_in_parent_directories(dotenv_dir):
    start = dotenv_dir / "child"
    start.mkdir()
    result = run_in(start, _RUN_MAIN.format(expr=f"os.environ.get({DOTENV_VAR!r})"))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "None", result.stdout
