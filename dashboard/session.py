"""Shared dashboard resources and page helpers."""

import os

import httpx
import streamlit as st

from dashboard.api_client import ApiError, ApiUnavailable, MedoraClient
from detection.config import API_KEY, API_URL

REPLAY_NOTICE = "Everything here uses recorded data, not a live network."
PREDICTION_NOTICE = "Alerts are leads to check, not confirmed attacks."


@st.cache_resource(max_entries=4)
def _build_client(url: str, api_key: str | None) -> MedoraClient:
    return MedoraClient(httpx.Client(base_url=url, timeout=30), api_key)


def get_client() -> MedoraClient:
    return _build_client(os.environ.get("MEDORA_API_URL", API_URL), API_KEY)


def show_api_problem(error: Exception) -> None:
    """Explain an API failure, then stop rendering the page."""
    if isinstance(error, ApiUnavailable):
        st.error(
            "Start it with `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`, then refresh this page.",
            title="Medora's backend isn't running",
            icon=":material/cloud_off:",
        )
    elif isinstance(error, ApiError):
        st.error(str(error), title="That request didn't work", icon=":material/error:")
    else:
        raise error
    st.stop()
