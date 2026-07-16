"""Provider-independent LLM errors."""


class LLMError(Exception):
    """An LLM provider failure with its original exception preserved."""

    def __init__(self, error: Exception) -> None:
        super().__init__(f"{type(error).__name__}: {error}")
        self.original_error = error
