"""Translators between wire formats and the internal format in `model_calling.core`."""


class TranslationError(ValueError):
    """A client or backend body couldn't be translated. Maps to HTTP 400 at ingress."""
