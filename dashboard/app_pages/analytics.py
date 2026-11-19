import streamlit as st

from dashboard import charts, views
from dashboard.api_client import ApiError, ApiUnavailable
from dashboard.session import get_client, show_api_problem

client = get_client()

SOURCES = {"replay": "Replay", "api": "API requests"}
source_col, bucket_col = st.columns(2)
source = source_col.segmented_control("Source", options=list(SOURCES), format_func=SOURCES.get, default="replay")
bucket = bucket_col.segmented_control(
    "Group by", options=["second", "minute"], format_func=str.capitalize, default="second"
)

try:
    statistics = client.statistics(source)
    timeline = client.timeline(source, bucket or "second")
    outcomes = client.label_outcomes(source)
except (ApiUnavailable, ApiError) as error:
    show_api_problem(error)

EMPTY_HINTS = {
    "replay": "Start a replay from Overview to see results here.",
    "api": "Records sent straight to the API show up here. Replays are counted separately.",
}

if not statistics["events_total"]:
    st.info(EMPTY_HINTS[source or "replay"], title="Nothing here yet", icon=":material/bar_chart_off:")
    st.stop()

with st.container(horizontal=True):
    st.metric("Records", f"{statistics['events_total']:,}", border=True)
    malicious_share = statistics["events_malicious"] / statistics["events_total"]
    st.metric("Looked like attacks", f"{malicious_share:.1%}", border=True)
    accuracy = views.outcome_accuracy(outcomes["by_class"])
    st.metric("Matched real labels", f"{accuracy:.2%}" if accuracy is not None else "No labels", border=True)

with st.container(border=True):
    st.subheader("Traffic over time")
    session, earlier_sessions = views.latest_session(timeline)
    frame = views.timeline_frame(session, bucket or "second")
    st.altair_chart(
        charts.timeline_area(frame),
        alt="Stacked area chart of predicted records for each class over processing time",
    )
    if earlier_sessions:
        start = frame["time"].min()
        st.caption(
            f"Showing the latest run, from {start:%Y-%m-%d %H:%M} UTC. {earlier_sessions:,} earlier "
            f"{'run is' if earlier_sessions == 1 else 'runs are'} counted in the totals above but not shown here."
        )
    st.caption(
        "Times show when Medora checked each record, not when the traffic happened, because the dataset has no "
        "real timestamps."
    )

if not outcomes["by_class"]:
    st.caption("Comparing with real labels needs labelled data, such as a replay.")
    st.stop()

matrix_col, severity_col = st.columns([3, 2])
with matrix_col.container(border=True):
    st.subheader("Predictions compared with real labels")
    st.altair_chart(
        charts.confusion_heatmap(views.outcome_matrix(outcomes["by_class"])),
        alt="Heatmap of actual class against predicted class for replayed records",
    )
    st.caption("Rows are the real type, and columns are what Medora predicted. Covers all replays so far.")

with severity_col.container(border=True):
    st.subheader("Are serious alerts usually right?")
    st.dataframe(
        views.severity_outcomes(outcomes["by_severity"]),
        hide_index=True,
        column_config={
            "Real attacks": st.column_config.NumberColumn(format="localized"),
            "Actually normal": st.column_config.NumberColumn(format="localized"),
            "Share real": st.column_config.NumberColumn(format="percent"),
        },
        alt="Share of alerted events that were real attacks for each severity",
    )
    st.caption(
        "Based on this data only, so it doesn't prove the risk scores are right."
    )
