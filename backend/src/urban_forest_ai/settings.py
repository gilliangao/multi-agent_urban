from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)


def get_secret(name: str) -> str | None:
    """Read a secret from env first, then Streamlit secrets if available."""

    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        return None

    return None


def get_model(name: str, default: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    return default
