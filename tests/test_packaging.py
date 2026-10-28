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
