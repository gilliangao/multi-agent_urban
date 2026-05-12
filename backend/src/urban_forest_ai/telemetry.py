from __future__ import annotations

import os


def optional_observe(*args, **kwargs):
    """Use Langfuse only when credentials are configured."""

    langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "0").lower() in {"1", "true", "yes"}
    if (
        langfuse_enabled
        and os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    ):
        from langfuse import observe

        return observe(*args, **kwargs)

    def decorator(func):
        return func

    return decorator
