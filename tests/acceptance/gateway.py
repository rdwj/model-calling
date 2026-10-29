"""
The gateway as `python -m model_calling` runs it, plus a callback that logs
each call's cost, for the acceptance test, which runs the gateway in its
own process.

Each call appends a JSON line to the file COSTS_ENV names: `{"cost": ...}`
when the backend reported `usage.cost` (OpenRouter's Messages endpoint
does, on the stream's message delta), otherwise `{"id": ...}` so the test
can look the cost up by response id.
"""

import json
import os

from model_calling import server
from model_calling.__main__ import main
from model_calling.callbacks import Callback
from model_calling.core import MessageDelta, MessageStart

COSTS_ENV = "MODEL_CALLING_ACCEPTANCE_COSTS"


def _cost(extras: dict) -> float | None:
    """`cost` from usage extras in any wire format."""
    for fields in extras.values():
        if fields.get("cost") is not None:
            return float(fields["cost"])
    return None


class CostLog(Callback):
    def __init__(self, path: str):
        self.path = path

    def _write(self, entry: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    async def on_request_end(self, model, messages, response, **kwargs):
        if isinstance(response, dict):  # a non-streaming call's native body
            cost = (response.get("usage") or {}).get("cost")
            self._write({"cost": float(cost)} if cost is not None else {"id": response.get("id")})

    async def on_stream_chunk(self, model, messages, chunk, **kwargs):
        if isinstance(chunk, MessageStart):
            self._write({"id": chunk.id})
        elif isinstance(chunk, MessageDelta) and chunk.usage is not None:
            cost = _cost(chunk.usage.extras)
            if cost is not None:
                self._write({"cost": cost})


if __name__ == "__main__":
    server.CLIENT.callbacks.append(CostLog(os.environ[COSTS_ENV]))
    main()
