"""
Checks that the offline suite is isolated from machine state and the network.
"""

import os
import re
from pathlib import Path

import httpx
import pytest

from tests.conftest import BACKEND_ENV_VARS


def test_home_is_a_scratch_directory():
    assert Path(os.path.expanduser("~")).name.startswith("model-calling-tests-home-")


def test_dotenv_loading_is_disabled():
    import dotenv

    assert dotenv.load_dotenv() is False


@pytest.mark.parametrize("var", BACKEND_ENV_VARS)
def test_backend_env_var_is_unset(var):
    assert var not in os.environ


@pytest.mark.parametrize("url", ["http://localhost:11434/api/tags", "https://openrouter.ai/api/v1"])
def test_sync_httpx_network_is_blocked(url):
    with pytest.raises(RuntimeError, match="over the network"):
        httpx.get(url)


async def test_async_httpx_network_is_blocked():
    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError, match="over the network"):
            await client.get("http://localhost:11434/api/tags")


# Absolute paths into a developer's home directory pass locally and break in
# every other checkout.
HOME_PATH = re.compile(r"""["'](/Users/|/home/|[A-Za-z]:\\\\Users\\\\)""")


def test_test_sources_have_no_home_paths():
    tests_dir = Path(__file__).parent
    offenders = [
        f"{path.relative_to(tests_dir)}:{n}"
        for path in sorted(tests_dir.rglob("*.py"))
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if HOME_PATH.search(line)
    ]
    assert not offenders, f"hardcoded home paths: {offenders}"
