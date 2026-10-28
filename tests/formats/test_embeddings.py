"""The OpenAI Embeddings translator: request validation, batching, and base64 output."""

import pytest
from openai import omit
from openai.lib._parsing._embeddings import parse_embedding_response
from openai.types import CreateEmbeddingResponse

from model_calling.core.embeddings import EmbeddingResponse
from model_calling.formats import TranslationError
from model_calling.formats import embeddings

BATCHES = {
    "string": ("hi", ["hi"]),
    "strings": (["a", "b"], ["a", "b"]),
    "token_array": ([1, 2, 3], [[1, 2, 3]]),
    "token_arrays": ([[1, 2], [3]], [[1, 2], [3]]),
}


@pytest.mark.parametrize("raw, batch", BATCHES.values(), ids=BATCHES.keys())
def test_input_becomes_a_batch(raw, batch):
    assert embeddings.request_to_internal({"model": "m", "input": raw}).input == batch


INVALID = {
    "not_an_object": ([], "JSON object"),
    "no_model": ({"input": "x"}, "model is required"),
    "no_input": ({"model": "m"}, "input is required"),
    "empty_list": ({"model": "m", "input": []}, "input must be"),
    "mixed": ({"model": "m", "input": ["a", 1]}, "input must be"),
    "empty_token_array": ({"model": "m", "input": [[1], []]}, "input must be"),
    "bool_tokens": ({"model": "m", "input": [True]}, "input must be"),
    "zero_dimensions": ({"model": "m", "input": "x", "dimensions": 0}, "dimensions"),
    "string_dimensions": ({"model": "m", "input": "x", "dimensions": "8"}, "dimensions"),
    "numeric_user": ({"model": "m", "input": "x", "user": 7}, "user"),
    "bad_encoding": ({"model": "m", "input": "x", "encoding_format": "int8"}, "encoding_format"),
}


@pytest.mark.parametrize("body, match", INVALID.values(), ids=INVALID.keys())
def test_invalid_request_is_a_translation_error(body, match):
    with pytest.raises(TranslationError, match=match):
        embeddings.request_to_internal(body)


def test_unknown_fields_are_kept_as_extras():
    req = embeddings.request_to_internal(
        {"model": "m", "input": "x", "user": "u", "truncate_prompt_tokens": 8}
    )
    assert (req.user, req.extras_for(embeddings.FORMAT)) == ("u", {"truncate_prompt_tokens": 8})


VECTORS = [[0.25, -1.5, 3.0], [1.0, 0.0, -0.125]]


@pytest.mark.parametrize("fmt", embeddings.ENCODING_FORMATS)
def test_response_round_trips_through_the_sdk(fmt):
    """Base64 vectors decode, with the SDK's own parser, to the floats we had."""
    internal = EmbeddingResponse(model="native", vectors=VECTORS, input_tokens=4)
    body = embeddings.response_from_internal(internal, "ollama/m", fmt)
    # The SDK builds responses unvalidated, which is how a base64 string reaches its decoder.
    parsed = CreateEmbeddingResponse.construct(**body)
    decoded = parse_embedding_response(parsed, encoding_format=omit)
    assert [d.embedding for d in decoded.data] == VECTORS
    assert (body["model"], body["usage"]) == ("ollama/m", {"prompt_tokens": 4, "total_tokens": 4})
    assert isinstance(body["data"][0]["embedding"], str) == (fmt == "base64")


def test_missing_usage_reports_zero_tokens():
    body = embeddings.response_from_internal(EmbeddingResponse(model="m", vectors=[[1.0]]), "m")
    assert body["usage"] == {"prompt_tokens": 0, "total_tokens": 0}
