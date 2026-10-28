"""The base for the library's typed response objects."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class WireModel(BaseModel):
    """A response object shaped as it travels on the wire.

    Unknown fields are kept, so a backend's additions (vLLM's `reasoning`)
    survive, and `to_dict` returns what was parsed, explicit nulls included.
    """

    model_config = ConfigDict(extra="allow")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_unset=True)
