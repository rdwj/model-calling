"""
Shared base for internal-format models.

`extras` carries wire fields the internal format doesn't model, keyed by the
format they came from (see `formats`). A translator for format X reads and
writes only `extras[X]`, so a same-format round trip is lossless while other
formats never see fields they can't interpret. Extras often hold explicit
nulls (Anthropic sends `"citations": null` on plain text blocks), so a
cross-format translator deciding whether foreign extras would be lost should
look only at non-null values.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

Extras = dict[str, dict[str, Any]]


class CoreModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extras: Extras = Field(default_factory=dict)

    def extras_for(self, fmt: str) -> dict[str, Any]:
        return self.extras.get(fmt, {})
