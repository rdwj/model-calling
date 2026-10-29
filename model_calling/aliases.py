"""
Model aliases: the names gateway clients use, mapped by a YAML file to a
backend and model.

    backends:
      local: {provider: ollama, base_url: http://ollama:11434}
      prod: {provider: vllm, base_url: https://vllm.example/v1, api_key_env: VLLM_API_KEY}
    models:
      chat: {backend: prod, model: gpt-oss-20b}
      embed: {backend: local, model: nomic-embed-text, dimensions: 768}

A backend names a provider and, optionally, its `base_url` and the
environment variable holding its key (never the key). A provider's name
works as a backend without a `backends` entry; it then uses the gateway's
defaults for that provider, environment overrides included. `AliasStore`
rereads the file when it changes, so an edit needs no restart.
"""

import base64
import binascii
import logging
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from model_calling.providers import PROVIDERS

logger = logging.getLogger(__name__)

CONFIG_ENV = "MODEL_CALLING_CONFIG"
DEFAULT_CONFIG_PATH = "model-calling.yaml"

_TOP_KEYS = {"backends", "models"}
_BACKEND_KEYS = {"provider", "base_url", "api_key_env"}
_MODEL_KEYS = {"backend", "model", "dimensions"}
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class AliasConfigError(ValueError):
    """The alias file is unreadable or invalid; the message names the key."""


@dataclass(frozen=True)
class Backend:
    provider: str
    base_url: str | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class Target:
    """What an alias resolves to, in the terms `Client` calls take."""

    alias: str
    model: str  # prefixed, as the library names it
    api_base: str | None
    api_key: str | None
    dimensions: int | None

    @property
    def provider(self) -> str:
        return self.model.split("/", 1)[0]


@dataclass(frozen=True)
class _Model:
    backend: Backend
    model: str
    dimensions: int | None


@dataclass(frozen=True)
class AliasConfig:
    models: Mapping[str, _Model]

    def resolve(self, alias: str, environ: Mapping[str, str] = os.environ) -> Target | None:
        entry = self.models.get(alias)
        if entry is None:
            return None
        backend = entry.backend
        api_key = environ.get(backend.api_key_env) if backend.api_key_env else None
        if backend.api_key_env and not api_key:
            raise AliasConfigError(
                f"the key variable {backend.api_key_env} for model {alias!r} is not set"
            )
        return Target(
            alias=alias,
            model=f"{backend.provider}/{entry.model}",
            api_base=backend.base_url,
            api_key=api_key,
            dimensions=entry.dimensions,
        )

    @property
    def aliases(self) -> list[str]:
        return list(self.models)


EMPTY = AliasConfig(models={})


def _mapping(value: Any, where: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AliasConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _check_keys(entry: dict, allowed: set, where: str) -> None:
    unknown = sorted(str(k) for k in entry if k not in allowed)
    if "api_key" in unknown:
        raise AliasConfigError(
            f"{where}.api_key: keys don't go in this file; put the key in an "
            "environment variable and name it in api_key_env"
        )
    if unknown:
        raise AliasConfigError(
            f"{where}: unknown key {unknown[0]!r} (expected {', '.join(sorted(allowed))})"
        )


def _string(entry: dict, key: str, where: str, required: bool) -> str | None:
    value = entry.get(key)
    if value is None:
        if required:
            raise AliasConfigError(f"{where}.{key} is required")
        return None
    if not isinstance(value, str) or not value.strip():
        raise AliasConfigError(f"{where}.{key} must be a non-empty string")
    return value.strip()


def _backend(name: str, entry: Any) -> Backend:
    where = f"backends.{name}"
    entry = _mapping(entry, where)
    _check_keys(entry, _BACKEND_KEYS, where)
    provider = _string(entry, "provider", where, required=True)
    if provider not in PROVIDERS:
        raise AliasConfigError(
            f"{where}.provider: unknown provider {provider!r} "
            f"(expected one of {', '.join(sorted(PROVIDERS))})"
        )
    base_url = _string(entry, "base_url", where, required=False)
    key_env = _string(entry, "api_key_env", where, required=False)
    if key_env is not None and not _ENV_NAME.match(key_env):
        # Not echoed: a value that isn't a variable name may be the key itself.
        raise AliasConfigError(f"{where}.api_key_env must name an environment variable")
    return Backend(provider, base_url.rstrip("/") if base_url else None, key_env)


def _model(alias: str, entry: Any, backends: Mapping[str, Backend]) -> _Model:
    where = f"models.{alias}"
    entry = _mapping(entry, where)
    _check_keys(entry, _MODEL_KEYS, where)
    backend_name = _string(entry, "backend", where, required=True)
    backend = backends.get(backend_name)
    if backend is None and backend_name in PROVIDERS:
        backend = Backend(backend_name)
    if backend is None:
        raise AliasConfigError(
            f"{where}.backend: unknown backend {backend_name!r} (define it under "
            f"backends, or use a provider: {', '.join(sorted(PROVIDERS))})"
        )
    dimensions = entry.get("dimensions")
    if dimensions is not None and (
        isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1
    ):
        raise AliasConfigError(f"{where}.dimensions must be a positive integer")
    return _Model(backend, _string(entry, "model", where, required=True), dimensions)


def parse(data: Any) -> AliasConfig:
    """Validate the loaded YAML document."""
    top = _mapping(data, "the file")
    _check_keys(top, _TOP_KEYS, "the file")
    backends = {
        str(name): _backend(str(name), entry)
        for name, entry in _mapping(top.get("backends"), "backends").items()
    }
    models = {}
    for alias, entry in _mapping(top.get("models"), "models").items():
        if not isinstance(alias, str) or not alias.strip():
            raise AliasConfigError(f"models: {alias!r} is not a usable model name")
        models[alias] = _model(alias, entry, backends)
    return AliasConfig(models=models)


class _UniqueKeyLoader:
    """A SafeLoader that refuses duplicate keys, which YAML would otherwise
    resolve silently to the last one. Built lazily: PyYAML is a server extra."""

    _cls = None

    @classmethod
    def get(cls):
        if cls._cls is None:
            import yaml

            class Loader(yaml.SafeLoader):
                def construct_mapping(self, node, deep=False):
                    seen = set()
                    for key_node, _ in node.value:
                        key = self.construct_object(key_node, deep=deep)
                        try:
                            hash(key)
                        except TypeError:
                            raise AliasConfigError(
                                f"line {key_node.start_mark.line + 1}: a key must be a name"
                            ) from None
                        if key in seen:
                            raise AliasConfigError(
                                f"line {key_node.start_mark.line + 1}: duplicate key {key!r}"
                            )
                        seen.add(key)
                    return super().construct_mapping(node, deep=deep)

            cls._cls = Loader
        return cls._cls


def dimension_mismatch(response: dict, target: Target) -> str | None:
    """Why an embeddings response breaks the alias's declared size, or None.
    Refusing it keeps a backend swap from putting mismatched vectors into an
    index."""
    if target.dimensions is None:
        return None
    for item in response.get("data", ()):
        size = _vector_size(item.get("embedding"))
        if size is not None and size != target.dimensions:
            return (
                f"model {target.alias!r} is declared with {target.dimensions} dimensions, "
                f"but its backend returned {size}"
            )
    return None


def _vector_size(embedding) -> int | None:
    if isinstance(embedding, list):
        return len(embedding)
    if isinstance(embedding, str):  # encoding_format=base64: packed float32s
        try:
            return len(base64.b64decode(embedding, validate=True)) // 4
        except (binascii.Error, ValueError):
            return None
    return None


def load(path: str) -> AliasConfig:
    """Read and validate the alias file at `path`."""
    import yaml

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.load(f, Loader=_UniqueKeyLoader.get())  # noqa: S506 - SafeLoader subclass
        return parse(data)
    except OSError as e:
        raise AliasConfigError(f"cannot read {path}: {e.strerror}") from e
    except UnicodeDecodeError as e:
        raise AliasConfigError(f"{path} is not UTF-8 text") from e
    except yaml.YAMLError as e:
        raise AliasConfigError(f"{path} is not valid YAML: {e}") from e
    except AliasConfigError as e:
        raise AliasConfigError(f"{path}: {e}") from e


class AliasStore:
    """The alias file's current contents, reloaded when the file changes.

    `current()` stats the file on every call (cheap, and it follows
    symlinks, so an OpenShift ConfigMap update counts). A file that fails to
    load leaves the last good config in place and is logged once per change.
    """

    def __init__(self, path: str):
        self.path = path
        self._config = EMPTY
        self._signature = None
        self._lock = threading.Lock()
        # Why the last change didn't load, or None.
        self.error: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "AliasStore":
        return cls(environ.get(CONFIG_ENV, "").strip() or DEFAULT_CONFIG_PATH)

    def _stat(self):
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return None
        return (st.st_mtime_ns, st.st_size, st.st_ino)

    def current(self) -> AliasConfig:
        signature = self._stat()
        if signature == self._signature:
            return self._config
        with self._lock:
            if signature == self._signature:
                return self._config
            if signature is None:
                if self._config is not EMPTY:
                    self.error = f"{self.path} is missing; keeping the last loaded models"
                    logger.error(self.error)
                self._signature = signature
                return self._config
            try:
                config = load(self.path)
            except Exception as e:  # any failure keeps serving the last good file
                if not isinstance(e, AliasConfigError):
                    logger.exception("unexpected error loading %s", self.path)
                self.error = f"{e}; keeping the last loaded models"
                logger.error(self.error)
            else:
                self._config, self.error = config, None
                logger.info("loaded %d models from %s", len(config.models), self.path)
            # Set after the config, so a reader that sees the new signature
            # without the lock also sees the config loaded for it.
            self._signature = signature
            return self._config

    def check(self) -> AliasConfig:
        """Load the file now, raising if it is present but invalid. For startup."""
        config = load(self.path) if self._stat() is not None else EMPTY
        with self._lock:
            self._config, self._signature, self.error = config, self._stat(), None
        return config
