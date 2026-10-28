"""
The exceptions the library raises, named after the `openai` SDK's.

Every error carries the provider that raised it, the backend's real HTTP
status (None when no usable response arrived), and the backend's raw error
body. The gateway decides what status its own client sees; see
`gateway.client_status`.
"""

from typing import Any


class APIError(Exception):
    """A request failed. Catch this to handle every error the library raises."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status: int | None = None,
        body: Any = None,
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status = status
        self.body = body


class APIConnectionError(APIError):
    """The backend could not be reached, or the connection broke."""


class APITimeoutError(APIConnectionError):
    """The backend did not answer in time."""


class APIResponseValidationError(APIError):
    """The backend answered with a body we could not read."""


class APIStatusError(APIError):
    """The backend answered with an error status. Subclasses name the common ones."""


class BadRequestError(APIStatusError):
    """HTTP 400, or a request the library itself rejects before sending."""


class UnsupportedParamsError(BadRequestError):
    """The backend can't honor these request parameters, and the caller did
    not ask to drop them. `params` names them."""

    def __init__(self, message: str, *, params: list[str], provider: str | None = None):
        super().__init__(message, provider=provider)
        self.params = params


class AuthenticationError(APIStatusError):
    """HTTP 401."""


class PermissionDeniedError(APIStatusError):
    """HTTP 403."""


class NotFoundError(APIStatusError):
    """HTTP 404."""


class RateLimitError(APIStatusError):
    """HTTP 429."""


class InternalServerError(APIStatusError):
    """HTTP 500 or above."""


BAD_REQUEST = 400
_BY_STATUS: dict[int, type[APIStatusError]] = {
    BAD_REQUEST: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    429: RateLimitError,
}
_SERVER_ERROR = 500


def error_for_status(
    status: int, message: str, *, provider: str | None = None, body: Any = None
) -> APIStatusError:
    """The `APIStatusError` subclass for an HTTP error status."""
    cls = _BY_STATUS.get(status)
    if cls is None:
        cls = InternalServerError if status >= _SERVER_ERROR else APIStatusError
    return cls(message, provider=provider, status=status, body=body)


__all__ = [
    "APIConnectionError",
    "APIError",
    "APIResponseValidationError",
    "APIStatusError",
    "APITimeoutError",
    "AuthenticationError",
    "BadRequestError",
    "InternalServerError",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
    "UnsupportedParamsError",
    "error_for_status",
]
