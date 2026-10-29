"""The base for the library's typed response objects."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Discriminator

# Tag for content blocks, deltas, and items a module doesn't model. They
# parse into a fallback model that keeps every field instead of failing.
UNKNOWN = "unknown"


class WireModel(BaseModel):
    """A response object shaped as it travels on the wire.

    Unknown fields are kept, so a backend's additions (vLLM's `reasoning`)
    survive, and `to_dict` returns what was parsed, explicit nulls included.
    """

    model_config = ConfigDict(extra="allow")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_unset=True)


def type_of(value: Any) -> Any:
    return value.get("type") if isinstance(value, dict) else getattr(value, "type", None)


def known_or_unknown(*known: str) -> Discriminator:
    """Discriminate on `type`, sending types not in `known` to the UNKNOWN tag."""
    return Discriminator(lambda value: type_of(value) if type_of(value) in known else UNKNOWN)
