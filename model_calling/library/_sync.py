"""
Run the async library on a private event loop, for the sync API.

One daemon thread per client owns the loop, so sync calls reuse pooled
connections the way async calls do, and work from code that already has a
running loop (a notebook) without nesting loops.

Shutdown cancels in-flight calls and closes open streams on the loop before
stopping it, so no caller waits on a loop that will never answer. It runs
on `close()` and, through `weakref.finalize`, at interpreter exit.
"""

import asyncio
import threading
import weakref
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from typing import Any, TypeVar

T = TypeVar("T")

# How long shutdown waits for the loop to cancel calls and close streams.
SHUTDOWN_TIMEOUT = 5.0


class ClosedError(RuntimeError):
    """A sync call on a client that has been closed."""


class _State:
    """What shutdown needs, kept apart from `LoopThread` so the finalizer
    doesn't keep the thread object alive."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, name="model-calling-sync", daemon=True
        )
        self.closed = False
        self.lock = threading.Lock()


class LoopThread:
    def __init__(self) -> None:
        self._state = _State()
        self._state.thread.start()
        self._finalizer = weakref.finalize(self, _shutdown, self._state, None)

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        state = self._state
        if state.closed:
            coro.close()
            raise ClosedError("this model-calling client is closed")
        if threading.current_thread() is state.thread:
            coro.close()
            raise RuntimeError(
                "a sync model-calling call was made from its own event loop "
                "(for example inside a callback); use the async API there"
            )
        future = asyncio.run_coroutine_threadsafe(coro, state.loop)
        try:
            return future.result()
        except BaseException:
            # A KeyboardInterrupt must not leave the call running unseen.
            future.cancel()
            raise

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
            # After shutdown the loop has already closed every open stream.
            if aclose is not None and not self._state.closed:
                self.run(aclose())

    def close(self, final: Callable[[], Coroutine[Any, Any, None]] | None = None) -> None:
        """Cancel in-flight calls, close open streams, run `final` (closing
        the connection pool), then stop the loop."""
        self._finalizer.detach()
        _shutdown(self._state, final)


def _shutdown(state: _State, final) -> None:
    with state.lock:
        if state.closed:
            return
        state.closed = True
    if state.loop.is_running():
        drain = asyncio.run_coroutine_threadsafe(_drain(final), state.loop)
        try:
            drain.result(SHUTDOWN_TIMEOUT)
        except Exception:
            drain.cancel()
        state.loop.call_soon_threadsafe(state.loop.stop)
        state.thread.join(SHUTDOWN_TIMEOUT)
    if not state.loop.is_running():
        state.loop.close()


async def _drain(final) -> None:
    current = asyncio.current_task()
    tasks = [t for t in asyncio.all_tasks() if t is not current]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.get_running_loop().shutdown_asyncgens()
    if final is not None:
        await final()


async def _anext(items: AsyncIterator[T]) -> T:
    return await items.__anext__()
