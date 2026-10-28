"""
The library API's cells of the (ingress, backend, scenario) grid: `Client`
calls against replayed backend fixtures. See tests/harness/grid.py.
"""

import pytest

from model_calling.errors import APIStatusError, UnsupportedParamsError
from tests.harness.grid import (
    LIBRARY_EMBEDDING_INGRESSES,
    LIBRARY_INGRESSES,
    Cell,
    embedding_grid,
    grid,
)
from tests.harness.replay import Exchange, fixture_path
from tests.harness.scenarios import (
    ALL_BACKENDS,
    BACKENDS,
    EMBEDDING_BACKENDS,
    EMBEDDING_INPUTS,
    EMBEDDING_SCENARIOS,
    SCENARIOS,
    refused_params,
)
from tests.harness.checks import COMPLETION_CHECKS, MESSAGE_CHECKS
from tests.harness.sdk import validate_embedding_response


def model(cell: Cell) -> str:
    return BACKENDS[cell.backend].model_for(SCENARIOS[cell.scenario])


async def completion(library, cell: Cell):
    body = SCENARIOS[cell.scenario].chat_completions_body(model(cell))
    return await library.client.acompletion(**body)


async def messages(library, cell: Cell):
    body = SCENARIOS[cell.scenario].messages_body(model(cell))
    return await library.client.amessages(**body)


async def check_backend_error(call, library, cell):
    recorded = Exchange.load(fixture_path(cell.backend, cell.scenario))
    with pytest.raises(APIStatusError) as err:
        await call(library, cell)
    assert err.value.status == recorded.status
    assert err.value.body == recorded.response_json


async def check_refused(call, library, cell):
    """Strict mode refuses what the backend can't honor, before calling it."""
    with pytest.raises(UnsupportedParamsError) as err:
        await call(library, cell)
    assert tuple(err.value.params) == refused_params(cell.backend, cell.scenario)


async def result(call, library, cell):
    """The call's result, with a stream collected into a list."""
    out = await call(library, cell)
    return [item async for item in out] if SCENARIOS[cell.scenario].stream else out


# --- library_embedding ----------------------------------------------------


def embedding_model(cell: Cell) -> str:
    return EMBEDDING_BACKENDS[cell.backend].model_for(EMBEDDING_SCENARIOS[cell.scenario])


async def embed(library, cell: Cell):
    body = EMBEDDING_SCENARIOS[cell.scenario].body(embedding_model(cell))
    return await library.client.aembedding(**body)


def recorded_vectors(cell: Cell) -> list:
    body = Exchange.load(fixture_path(cell.backend, cell.scenario)).response_json
    return body["embeddings"] if "embeddings" in body else [d["embedding"] for d in body["data"]]


async def check_embedding(library, cell):
    resp = await embed(library, cell)
    validate_embedding_response(resp.to_dict())
    assert [d.embedding for d in resp.data] == recorded_vectors(cell)
    assert [d.index for d in resp.data] == list(range(len(EMBEDDING_INPUTS)))
    assert resp.model == embedding_model(cell)
    assert resp.usage.prompt_tokens > 0


async def check_embedding_error(library, cell):
    await check_backend_error(embed, library, cell)


EMBEDDING_CHECKS = {"embedding": check_embedding, "embedding_error": check_embedding_error}
# Ingress -> (the library call, its result checks).
CALLS = {
    "library_completion": (completion, COMPLETION_CHECKS),
    "library_messages": (messages, MESSAGE_CHECKS),
}


def assert_auth(cell: Cell, replay):
    """A backend with a key gets it as a bearer token; one without gets none."""
    assert replay.requests, "client never called the backend"
    key = ALL_BACKENDS[cell.backend].gateway_config.get("api_key")
    assert replay.requests[0].headers.get("authorization") == (f"Bearer {key}" if key else None)


@pytest.mark.parametrize(
    "cell", grid(ingresses=LIBRARY_INGRESSES) + embedding_grid(LIBRARY_EMBEDDING_INGRESSES)
)
async def test_cell(library, cell: Cell):
    replay = library.use(cell.backend, cell.scenario)
    if cell.ingress in LIBRARY_EMBEDDING_INGRESSES:
        await EMBEDDING_CHECKS[cell.scenario](library, cell)
    elif refused_params(cell.backend, cell.scenario):
        await check_refused(CALLS[cell.ingress][0], library, cell)
        assert not replay.requests, "a refused request reached the backend"
        return
    elif cell.scenario == "backend_error":
        await check_backend_error(CALLS[cell.ingress][0], library, cell)
    else:
        call, checks = CALLS[cell.ingress]
        checks[cell.scenario](await result(call, library, cell), model(cell))
    assert_auth(cell, replay)
