"""Medora dashboard. Run from the project root: python -m streamlit run dashboard/app.py"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.session import PREDICTION_NOTICE, REPLAY_NOTICE  # noqa: E402

st.set_page_config(page_title="Medora", page_icon=":material/shield:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/monitoring:", default=True),
        st.Page("app_pages/alerts.py", title="Alerts", icon=":material/notifications:"),
        st.Page("app_pages/analytics.py", title="Analytics", icon=":material/bar_chart:"),
        st.Page("app_pages/model_performance.py", title="Model performance", icon=":material/model_training:"),
    ]
)

with st.sidebar:
    st.markdown("**Medora**")
    st.caption("Security monitoring for medical IoT networks")
    st.badge("Demo with recorded data", icon=":material/history:", color="blue")
    st.caption(REPLAY_NOTICE)
    st.caption(PREDICTION_NOTICE)

st.title(page.title)
page.run()
