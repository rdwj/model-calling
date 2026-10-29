"""
The synchronous HTTP client for a running gateway, `SyncModelCallingClient`.
"""

import threading

import httpx

from model_calling.client import SyncModelCallingClient

MODELS = {"object": "list", "data": [{"id": "ollama/qwen3:4b", "object": "model"}]}


def test_sync_client_runs_without_a_current_event_loop():
    """It worked only where asyncio.get_event_loop() could make a loop: not
    in a worker thread, and not at all on Python 3.14."""
    seen, errors = [], []

    def run():
        try:
            client = SyncModelCallingClient("http://gateway.test")
            client.async_client.client = httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json=MODELS))
            )
            try:
                seen.append(client.list_models())
            finally:
                client.close()
            assert client.loop.is_closed()
        except Exception as e:  # reported to the test's thread below
            errors.append(e)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    assert not errors, errors
    assert seen == [MODELS]
