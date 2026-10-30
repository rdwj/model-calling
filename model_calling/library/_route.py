"""
Where a `Client` call goes and with what settings: the per-call overrides
merged over the prefix's config, and the provider built from the result.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from model_calling.callbacks import Callback
from model_calling.errors import BadRequestError
from model_calling.providers import (
    PROVIDERS,
    HttpClients,
    Provider,
    merge_headers,
    provider_for,
    split_model,
)


@dataclass(frozen=True)
class CallOptions:
    """The keyword arguments every call method takes besides request fields."""

    api_base: str | None = None
    api_key: str | None = None
    drop_params: bool | None = None
    timeout: float | None = None
    extra_headers: Mapping[str, str] | None = None


def same_base(a: str, b: str | None) -> bool:
    return b is not None and a.rstrip("/") == b.rstrip("/")


def call_config(
    config: Mapping[str, Any], callbacks: Sequence[Callback], opts: CallOptions
) -> dict[str, Any]:
    """The prefix's `config` with `opts` applied, as `provider_for` reads it.

    The `organization` setting identifies an account with the configured
    backend, so a per-call `api_base` elsewhere leaves it off. Headers the
    user configured go everywhere, and `extra_headers` go over all of them.
    """
    config = dict(config)
    if opts.api_base is not None:
        if not same_base(opts.api_base, config.get("base_url")):
            config.pop("organization", None)
        config["base_url"] = opts.api_base
    if opts.api_key is not None:
        config["api_key"] = opts.api_key
    if opts.timeout is not None:
        config["timeout"] = opts.timeout
    if opts.extra_headers:
        config["headers"] = merge_headers(config.get("headers"), opts.extra_headers)
    config["callbacks"] = [*callbacks, *config.get("callbacks", ())]
    return config


def build_provider(
    configs: Mapping[str, Mapping[str, Any]],
    callbacks: Sequence[Callback],
    http: HttpClients,
    model: str,
    opts: CallOptions,
) -> Provider:
    """The provider for `model` under `configs` (the client's), with `opts`."""
    prefix, _ = split_model(model)
    if prefix not in PROVIDERS:
        raise BadRequestError(
            f"no provider for {model!r}: the library supports the prefixes "
            f"{', '.join(sorted(PROVIDERS))}"
        )
    config = call_config(configs.get(prefix, {}), callbacks, opts)
    try:
        return provider_for(model, {prefix: config}, http)
    except ValueError as e:
        raise BadRequestError(str(e), provider=prefix) from e
