"""Errors raised by the web application's domain and service layers."""


class DomainError(Exception):
    """Failure with a message safe to show to the user."""
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

class ValidationError(DomainError):
    """The input is not acceptable."""
    pass

class NotFoundError(DomainError):
    """The requested folder, video, or task does not exist."""
    pass

class ConflictError(DomainError):
    """The request clashes with the current state of the library."""
    pass

class MediaError(ValidationError):
    """A media file could not be read."""
    pass

class JobCancelled(Exception):
    """Raised inside a running job after the user cancelled it."""
    pass
