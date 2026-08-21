class DomainError(Exception):
    """Base for all domain errors."""


class NotFound(DomainError):  # noqa: N818
    pass


class ValidationError(DomainError):
    pass


class RateLimited(DomainError):  # noqa: N818
    pass
