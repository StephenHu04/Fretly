"""Base error shared by every pipeline stage."""

from __future__ import annotations


class PipelineError(Exception):
    """An expected, user-facing failure.

    ``code`` is stable for programmatic handling; ``message`` is safe to show
    to a user and never includes internal paths, stderr or stack traces.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
