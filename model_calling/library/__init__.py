"""
Module-level conveniences over a default `Client`, created on first use.

They take the same arguments as the `Client` methods. Use a `Client` of your
own for custom config or callbacks, or to control when connections close.
"""

import threading

from model_calling.library.client import Client

_default: Client | None = None
_default_lock = threading.Lock()


def default_client() -> Client:
    global _default
    with _default_lock:
        if _default is None:
            _default = Client()
        return _default


def completion(*args, **kwargs):
    return default_client().completion(*args, **kwargs)


async def acompletion(*args, **kwargs):
    return await default_client().acompletion(*args, **kwargs)


def messages(*args, **kwargs):
    return default_client().messages(*args, **kwargs)


async def amessages(*args, **kwargs):
    return await default_client().amessages(*args, **kwargs)


__all__ = [
    "Client",
    "acompletion",
    "amessages",
    "completion",
    "default_client",
    "messages",
]
