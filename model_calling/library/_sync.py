"""
Run the async library on a private event loop, for the sync API.

One daemon thread per client owns the loop, so sync calls reuse pooled
connections the way async calls do, and work from code that already has a
running loop (a notebook) without nesting loops.
"""

import asyncio
import threading
from collections.abc import AsyncIterator, Coroutine, Iterator
from typing import Any, TypeVar

T = TypeVar("T")


class LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self.loop.run_forever, name="model-calling-sync", daemon=True
        )
        self._thread.start()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        if threading.current_thread() is self._thread:
            coro.close()
            raise RuntimeError(
                "a sync model-calling call was made from its own event loop "
                "(for example inside a callback); use the async API there"
            )
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def iterate(self, items: AsyncIterator[T]) -> Iterator[T]:
        """Drive an async iterator from sync code, closing it however iteration ends."""
        try:
            while True:
                try:
                    item = self.run(_anext(items))
                except StopAsyncIteration:
                    return
                yield item
        finally:
            aclose = getattr(items, "aclose", None)
            if aclose is not None:
                self.run(aclose())

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join()
        self.loop.close()


async def _anext(items: AsyncIterator[T]) -> T:
    return await items.__anext__()
