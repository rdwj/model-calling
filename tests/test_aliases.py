"""
The alias file: validation, resolution, and reloading on change.
"""

import os
import textwrap

import pytest
import yaml

from model_calling.aliases import (
    CONFIG_ENV,
    DEFAULT_CONFIG_PATH,
    AliasConfigError,
    AliasStore,
    Target,
    load,
    parse,
)

KEY_ENV = "MODEL_CALLING_TEST_BACKEND_KEY"

GOOD = {
    "backends": {
        "local": {"provider": "ollama", "base_url": "http://ollama:11434/"},
        "prod": {
            "provider": "vllm",
            "base_url": "https://vllm.example/v1",
            "api_key_env": KEY_ENV,
        },
    },
    "models": {
        "chat": {"backend": "prod", "model": "gpt-oss-20b"},
        "embed": {"backend": "local", "model": "nomic-embed-text", "dimensions": 768},
        # A provider works as a backend with the gateway's defaults.
        "small": {"backend": "ollama", "model": "qwen3:4b"},
        # Aliases are free-form, so an app can keep the name vLLM serves.
        "openai/gpt-oss-20b": {"backend": "prod", "model": "gpt-oss-20b"},
    },
}


@pytest.mark.parametrize(
    "alias, environ, expected",
    [
        (
            "chat",
            {KEY_ENV: "k"},
            Target("chat", "vllm/gpt-oss-20b", "https://vllm.example/v1", "k", None),
        ),
        (
            "embed",
            {},
            Target("embed", "ollama/nomic-embed-text", "http://ollama:11434", None, 768),
        ),
        ("small", {}, Target("small", "ollama/qwen3:4b", None, None, None)),
        (
            "openai/gpt-oss-20b",
            {KEY_ENV: "k"},
            Target("openai/gpt-oss-20b", "vllm/gpt-oss-20b", "https://vllm.example/v1", "k", None),
        ),
        ("missing", {}, None),
    ],
)
def test_resolve(alias, environ, expected):
    assert parse(GOOD).resolve(alias, environ) == expected


def test_resolve_refuses_a_backend_whose_key_variable_is_unset():
    with pytest.raises(AliasConfigError, match=f"{KEY_ENV} for model 'chat' is not set"):
        parse(GOOD).resolve("chat", {})


def test_aliases_keep_file_order():
    assert parse(GOOD).aliases == list(GOOD["models"])


def test_an_empty_document_has_no_models():
    assert parse(None).aliases == []
    assert parse({"models": None}).aliases == []


def with_model(**entry):
    return {"models": {"chat": entry}}


INVALID = {
    "not_a_mapping": (["chat"], "the file: expected a mapping"),
    "unknown_top_key": ({"model": {}}, "the file: unknown key 'model'"),
    "unknown_model_key": (
        with_model(backend="ollama", model="m", temperature=1),
        "models.chat: unknown key 'temperature'",
    ),
    "model_missing": (with_model(backend="ollama"), "models.chat.model is required"),
    "backend_missing": (with_model(model="m"), "models.chat.backend is required"),
    "unknown_backend": (
        with_model(backend="nowhere", model="m"),
        "models.chat.backend: unknown backend 'nowhere'",
    ),
    "cohere_not_ported": (
        with_model(backend="cohere", model="command-a"),
        "models.chat.backend: unknown backend 'cohere'",
    ),
    "empty_model": (with_model(backend="ollama", model=" "), "models.chat.model must be"),
    "dimensions_zero": (
        with_model(backend="ollama", model="m", dimensions=0),
        "models.chat.dimensions must be a positive integer",
    ),
    "dimensions_bool": (
        with_model(backend="ollama", model="m", dimensions=True),
        "models.chat.dimensions must be a positive integer",
    ),
    "unknown_provider": (
        {"backends": {"b": {"provider": "cohere"}}},
        "backends.b.provider: unknown provider 'cohere'",
    ),
    "key_literal": (
        {"backends": {"b": {"provider": "vllm", "api_key": "sk-123"}}},
        "backends.b.api_key: keys don't go in this file",
    ),
    "key_env_not_a_name": (
        {"backends": {"b": {"provider": "vllm", "api_key_env": "sk-123"}}},
        "backends.b.api_key_env must name an environment variable",
    ),
    "backend_not_mapping": ({"backends": {"b": "vllm"}}, "backends.b: expected a mapping"),
}


@pytest.mark.parametrize("document, message", INVALID.values(), ids=INVALID)
def test_parse_names_the_offending_key(document, message):
    with pytest.raises(AliasConfigError, match=f"^{message}") as err:
        parse(document)
    assert "sk-123" not in str(err.value)


def write(path, text: str) -> None:
    path.write_text(textwrap.dedent(text))


@pytest.mark.parametrize(
    "text, message",
    [
        (
            "models:\n  chat: {backend: ollama, model: a}\n  chat: {backend: ollama, model: b}\n",
            "line 3: duplicate key 'chat'",
        ),
        ("models: [unclosed\n", "is not valid YAML"),
        ("models:\n  chat: {backend: nowhere, model: m}\n", "unknown backend 'nowhere'"),
    ],
    ids=["duplicate", "syntax", "invalid"],
)
def test_load_names_the_file(tmp_path, text, message):
    path = tmp_path / "models.yaml"
    write(path, text)
    with pytest.raises(AliasConfigError, match=message) as err:
        load(str(path))
    assert str(path) in str(err.value)


def test_load_refuses_yaml_tags_that_build_objects(tmp_path):
    path = tmp_path / "models.yaml"
    write(path, "models: !!python/object/apply:os.system ['true']\n")
    with pytest.raises(AliasConfigError, match="is not valid YAML"):
        load(str(path))


@pytest.mark.parametrize(
    "environ, path",
    [({}, DEFAULT_CONFIG_PATH), ({CONFIG_ENV: " /etc/mc/models.yaml "}, "/etc/mc/models.yaml")],
)
def test_store_path_from_env(environ, path):
    assert AliasStore.from_env(environ).path == path


class Edits:
    """Rewrite a file as editors and ConfigMap updates do: a new file
    renamed into place, with a distinct mtime each time."""

    def __init__(self, path):
        self.path = path
        self.mtime_ns = 0

    def __call__(self, document) -> None:
        staged = self.path.with_suffix(".tmp")
        text = document if isinstance(document, str) else yaml.safe_dump(document)
        staged.write_text(text)
        self.mtime_ns = max(self.mtime_ns + 1_000_000, os.stat(staged).st_mtime_ns)
        os.utime(staged, ns=(self.mtime_ns, self.mtime_ns))
        staged.replace(self.path)


@pytest.fixture
def edits(tmp_path):
    return Edits(tmp_path / "models.yaml")


def models(model: str) -> dict:
    return {"models": {"chat": {"backend": "ollama", "model": model}}}


def target_model(store: AliasStore) -> str | None:
    target = store.current().resolve("chat")
    return target and target.model


def test_store_without_a_file_is_empty(edits):
    store = AliasStore(str(edits.path))
    assert store.current().aliases == []
    assert store.error is None


def test_store_picks_up_a_new_file_and_each_change(edits):
    store = AliasStore(str(edits.path))
    assert target_model(store) is None
    edits(models("a"))
    assert target_model(store) == "ollama/a"
    edits(models("b"))
    assert target_model(store) == "ollama/b"


def test_store_keeps_the_last_good_config_and_logs_once(edits, caplog):
    store = AliasStore(str(edits.path))
    edits(models("a"))
    store.current()
    edits("models: [unclosed\n")
    for _ in range(3):
        assert target_model(store) == "ollama/a"
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "keeping the last loaded models" in errors[0].message
    assert "is not valid YAML" in store.error
    edits(models("b"))
    assert target_model(store) == "ollama/b"
    assert store.error is None


def test_store_keeps_the_config_when_the_file_disappears(edits):
    store = AliasStore(str(edits.path))
    edits(models("a"))
    store.current()
    edits.path.unlink()
    assert target_model(store) == "ollama/a"
    assert "is missing" in store.error


def test_store_follows_a_symlink_swap(tmp_path):
    """A mounted ConfigMap updates by repointing a symlink, not by writing
    the file the app opens."""
    first, second = tmp_path / "v1", tmp_path / "v2"
    first.mkdir(), second.mkdir()
    (first / "models.yaml").write_text(yaml.safe_dump(models("a")))
    (second / "models.yaml").write_text(yaml.safe_dump(models("b") | {"backends": {}}))
    data = tmp_path / "..data"
    data.symlink_to(first)
    link = tmp_path / "models.yaml"
    link.symlink_to(data / "models.yaml")
    store = AliasStore(str(link))
    assert target_model(store) == "ollama/a"
    staged = tmp_path / "..data_tmp"
    staged.symlink_to(second)
    staged.replace(data)
    assert target_model(store) == "ollama/b"


def test_check_raises_for_an_invalid_file(edits):
    edits("models: [unclosed\n")
    with pytest.raises(AliasConfigError):
        AliasStore(str(edits.path)).check()


def test_check_loads_a_good_file(edits):
    edits(models("a"))
    store = AliasStore(str(edits.path))
    assert store.check().aliases == ["chat"]
    assert target_model(store) == "ollama/a"
