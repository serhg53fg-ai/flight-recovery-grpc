"""Errors shared across inference adapters and the gRPC boundary."""


class InvalidInferenceRequest(ValueError):
    """The caller omitted or supplied invalid backend-required context."""
