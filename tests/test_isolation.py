"""
Checks that the offline suite is isolated from machine state and the network.
"""

import os
from pathlib import Path

import httpx
import pytest

from tests.conftest import BACKEND_ENV_VARS


def test_home_is_a_scratch_directory():
    assert Path(os.path.expanduser("~")).name.startswith("model-calling-tests-home-")


def test_capability_cache_lives_in_tmp_path(tmp_path):
    from model_calling.capability_cache import capability_cache

    assert Path(capability_cache.cache_file).parent == tmp_path
    assert capability_cache.cache["models"] == {}


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
