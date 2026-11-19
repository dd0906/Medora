import streamlit as st

from dashboard import charts, views
from dashboard.api_client import ApiError, ApiUnavailable
from dashboard.session import get_client, show_api_problem

client = get_client()

REPLAY_INPUTS = ("replay_limit", "replay_rate")

st.info(
    "Medora is checking recorded traffic from the CICIoMT2024 dataset, not a live network.",
    title="Demo mode",
    icon=":material/history:",
)


def reset_inputs() -> None:
    for key in REPLAY_INPUTS:
        st.session_state[key] = None


with st.container(border=True):
    st.subheader("Replay recorded traffic")
    limit_col, rate_col, start_col, reset_col = st.columns([2, 2, 1, 1], vertical_alignment="bottom")
    limit = limit_col.number_input(
        "How many records", min_value=100, max_value=103_889, value=None, step=500,
        placeholder="100 to 103,889", key="replay_limit",
    )
    rate = rate_col.number_input(
        "Records per second", min_value=50, max_value=5000, value=None, step=50,
        placeholder="50 to 5,000", key="replay_rate",
    )
    ready = limit is not None and rate is not None
    if start_col.button("Start", icon=":material/play_arrow:", type="primary", width="stretch", disabled=not ready):
        try:
            client.start_replay(int(limit), float(rate))
        except (ApiUnavailable, ApiError) as error:
            show_api_problem(error)
    reset_col.button("Reset", icon=":material/restart_alt:", width="stretch", on_click=reset_inputs)
    st.caption("Not sure what to enter? Try 1,000 records at 200 per second. It takes about 5 seconds.")


@st.fragment(run_every="2s")
def live_view() -> None:
    try:
        replay = client.replay_status()
        statistics = client.statistics()
        events = client.events(limit=15)
    except (ApiUnavailable, ApiError) as error:
        show_api_problem(error)

    if replay["state"] == "running":
        total = replay["total"] or 1
        # The total is unknown until the records have loaded.
        text = f"Checking {replay['processed']:,} of {replay['total']:,} records" if replay["total"] else "Loading records…"
        st.progress(min(replay["processed"] / total, 1.0), text=text)
        if st.button("Stop replay", icon=":material/stop:"):
            client.stop_replay()
    elif replay["state"] == "failed":
        st.error(replay["error"], title="The replay stopped with an error", icon=":material/error:")
    elif replay["state"] in ("finished", "stopped"):
        scored = replay["processed"] - replay["rejected"]
        accuracy = replay["correct"] / scored if scored else 0
        st.success(
            f"Checked {replay['processed']:,} records. {accuracy:.1%} matched the real labels, "
            f"and {replay['new_alerts']:,} new {'alert was' if replay['new_alerts'] == 1 else 'alerts were'} opened.",
            title="Replay finished" if replay["state"] == "finished" else "Replay stopped",
            icon=":material/task_alt:",
        )

    with st.container(horizontal=True):
        st.metric("Records checked", f"{statistics['events_total']:,}", border=True)
        st.metric("Looked normal", f"{statistics['events_benign']:,}", border=True)
        st.metric("Looked like attacks", f"{statistics['events_malicious']:,}", border=True)
        st.metric("Open alerts", f"{views.open_alert_count(statistics):,}", border=True)

    if not statistics["events_total"]:
        st.info(
            "Start a replay above to see results here.",
            title="Nothing checked yet",
            icon=":material/play_circle:",
        )
        return

    class_col, severity_col = st.columns(2)
    with class_col.container(border=True):
        st.subheader("Traffic by type")
        st.altair_chart(
            charts.class_bars(views.class_counts(statistics)),
            alt="Bar chart of records predicted for each traffic class",
        )
    with severity_col.container(border=True):
        st.subheader("Alerts by severity")
        st.altair_chart(
            charts.severity_bars(views.severity_counts(statistics)),
            alt="Bar chart of alert counts for each severity",
        )

    with st.container(border=True):
        st.subheader("Latest results")
        st.dataframe(
            views.events_table(events),
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "Risk": st.column_config.NumberColumn(format="%.1f"),
            },
            alt="The most recent detection events",
        )


live_view()

with st.expander("Delete replay results", icon=":material/delete:"):
    st.caption("Removes all replay results and their alerts. This can't be undone.")
    with st.form("clear_replay_data", border=False, clear_on_submit=True):
        confirmed = st.checkbox("Yes, delete them")
        submitted = st.form_submit_button("Delete")
    if submitted and not confirmed:
        st.warning("Tick the box first.", icon=":material/warning:")
    elif submitted:
        try:
            client.clear_replay_data()
        except ApiError as error:
            st.warning(str(error), icon=":material/warning:")
        except ApiUnavailable as error:
            show_api_problem(error)
        else:
            st.toast("Replay results deleted", icon=":material/check:")
