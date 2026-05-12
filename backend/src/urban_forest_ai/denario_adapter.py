from __future__ import annotations

from functools import lru_cache
from pathlib import Path


DENARIO_PROMPT_SNIPPET = (
    "Your goal is to generate a groundbreaking idea for a scientific paper. "
    "Generate an original idea given the data description."
)


@lru_cache(maxsize=1)
def detect_denario_runtime() -> dict[str, str]:
    """Return integration status for Denario and AG2 without crashing the app."""

    status = {
        "denario_dependency": "unknown",
        "ag2_dependency": "unknown",
        "denario_runtime": "unverified",
        "ag2_runtime": "unverified",
        "active_runtime": "internal",
        "fallback_mode": "enabled",
    }

    try:
        import denario  # noqa: F401

        status["denario_dependency"] = "installed"
        status["denario_runtime"] = "available"
    except Exception as exc:  # pragma: no cover - diagnostic path
        status["denario_dependency"] = "unavailable"
        status["denario_runtime"] = f"blocked: {type(exc).__name__}: {exc}"

    try:
        from autogen import AssistantAgent  # noqa: F401

        status["ag2_dependency"] = "installed"
        status["ag2_runtime"] = "available"
    except Exception as exc:  # pragma: no cover - diagnostic path
        status["ag2_dependency"] = "unavailable"
        status["ag2_runtime"] = f"blocked: {type(exc).__name__}: {exc}"

    if status["denario_runtime"] == "available" and status["ag2_runtime"] == "available":
        status["active_runtime"] = "denario_ag2"
        status["fallback_mode"] = "disabled"

    return status


def load_denario_prompt_hint() -> str:
    """Read a small Denario-authored prompt file if available."""

    prompt_path = (
        Path(".venv312")
        / "lib"
        / "python3.12"
        / "site-packages"
        / "denario"
        / "langgraph_agents"
        / "prompts.py"
    )
    if not prompt_path.exists():
        return DENARIO_PROMPT_SNIPPET

    text = prompt_path.read_text()
    marker = "Your goal is to generate a groundbreaking idea for a scientific paper."
    if marker in text:
        return marker
    return DENARIO_PROMPT_SNIPPET
