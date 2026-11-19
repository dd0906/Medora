import pandas as pd
import streamlit as st

from dashboard import charts, views
from dashboard.api_client import ApiError, ApiUnavailable
from dashboard.session import get_client, show_api_problem
from detection.config import CLASSES
from detection.risk_scoring import SEVERITIES

client = get_client()

status = st.segmented_control(
    "Status",
    options=list(views.STATUS_LABELS),
    format_func=views.STATUS_LABELS.get,
    key="alert_status_filter",
)
severity_col, class_col, source_col = st.columns(3)
severity = severity_col.selectbox("Severity", ["All", *reversed(SEVERITIES[1:])])
predicted = class_col.selectbox("Attack type", ["All", *CLASSES[1:]], format_func=views.class_label)
source = source_col.selectbox("Source", ["All", *views.SOURCE_LABELS], format_func=lambda name: views.SOURCE_LABELS.get(name, name))

try:
    alerts = client.alerts(
        status=status,
        severity=None if severity == "All" else severity,
        predicted_class=None if predicted == "All" else predicted,
        source=None if source == "All" else source,
        limit=200,
    )
except (ApiUnavailable, ApiError) as error:
    show_api_problem(error)

if not alerts:
    st.info(
        "Run a replay from Overview, or change the filters.",
        title="No alerts here",
        icon=":material/notifications_off:",
    )
    st.stop()

selection = st.dataframe(
    views.alerts_table(alerts),
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="alert_table",
    column_config={
        "Risk": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
        "Events": st.column_config.NumberColumn(format="localized"),
    },
    alt="Alerts matching the filters, highest risk first",
)
selected_rows = selection.selection.rows
if not selected_rows:
    st.caption("Tick a row to see the details.")
    st.stop()

alert = alerts[selected_rows[0]]
explanation = alert["explanation"]
is_rule_alert = alert["detector"] == "arp_rules"
is_baseline_alert = alert["detector"] == "device_baseline"

with st.container(border=True):
    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader(views.alert_title(alert))
        st.badge(
            alert["severity"],
            icon=views.SEVERITY_ICONS[alert["severity"]],
            color=views.SEVERITY_BADGE_COLORS[alert["severity"]],
        )
        st.badge(views.STATUS_LABELS[alert["status"]], color="gray")
        st.badge(views.SOURCE_LABELS[alert["source"]], color="blue")

    summary, status_note = views.alert_summary(alert)
    st.write(summary)
    st.caption(status_note)

    with st.container(horizontal=True):
        st.metric("Risk score", f"{alert['risk_score']:.1f}", border=True)
        if alert["max_confidence"] is not None:
            st.metric("Model confidence", f"{alert['max_confidence']:.1%}", border=True)
        count_label = "Times seen" if is_rule_alert else "Minutes flagged" if is_baseline_alert else "Events"
        st.metric(count_label, f"{alert['event_count']:,}", border=True)

    actions = views.available_actions(alert["status"])
    if actions:
        with st.container(horizontal=True):
            for action in actions:
                if st.button(views.ACTION_LABELS[action], key=f"action_{action}"):
                    try:
                        client.update_alert(alert["id"], action)
                    except ApiError as error:
                        st.warning(str(error), icon=":material/warning:")
                    except ApiUnavailable as error:
                        show_api_problem(error)
                    else:
                        st.rerun()
    else:
        st.caption("This alert is closed. New detections will open a new one.")

if is_rule_alert:
    with st.container(border=True):
        st.subheader("What the rule saw")
        st.dataframe(
            views.rule_evidence_table(explanation),
            hide_index=True,
            alt="Evidence the ARP rule found in the capture",
        )
        st.caption(
            "Rule alerts stop at Medium risk, because these rules also flagged some normal captures. "
            "To check one, compare the MAC address with your list of devices."
        )
    st.stop()

if is_baseline_alert:
    with st.container(border=True):
        st.subheader("Compared with this device's normal behaviour")
        st.write(explanation["detail"])
        evidence = views.baseline_evidence_table(explanation)
        if not evidence.empty:
            st.dataframe(evidence, hide_index=True, alt="Measures that went above this device's own limits")
        if explanation["limited_history"]:
            st.warning("This device has little history, so its limits are less reliable.", icon=":material/warning:")
        st.caption(
            f"Capture: {explanation['capture']}. These alerts stay at Low risk, because they can also flag normal "
            "traffic, and devices that were only affected by an attack."
        )
    st.stop()

with st.container(border=True):
    st.subheader("Why the model flagged this")
    features = explanation["influential_features"]
    st.write(views.reason_sentence(alert["predicted_class"], features))
    if features:
        effects = views.effects_frame(features)
        st.altair_chart(charts.effect_bars(effects), alt="How much each feature pushed the model towards this answer")
        st.caption(
            "Longer bars pushed the model harder towards this answer. This explains the model's decision, not what "
            "caused the traffic. Feature descriptions come from the column names, because the dataset doesn't define them."
        )
        with st.expander("Show the numbers"):
            st.dataframe(
                effects,
                hide_index=True,
                column_config={
                    "Push": st.column_config.NumberColumn("Push (points)", format="+%.1f"),
                    "Value": st.column_config.NumberColumn(format="%.3f"),
                    "Column": st.column_config.TextColumn("Dataset column"),
                },
                alt="Feature effects for the record that opened this alert",
            )

with st.expander("How the risk score was worked out"):
    st.dataframe(
        views.risk_components_table(explanation["risk_components_when_opened"]),
        hide_index=True,
        alt="Parts of the risk score when the alert opened",
    )
    st.caption("When the alert opened. The score can rise as more events join it. Confidence shows how sure the model was, not how serious the attack is.")

try:
    outcomes = client.alert_outcomes(alert["id"])
except (ApiUnavailable, ApiError) as error:
    show_api_problem(error)

if outcomes:
    with st.container(border=True):
        st.subheader("What these events really were")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Real label": views.class_label(name), "Events": count, "Share": count / sum(outcomes.values())}
                    for name, count in sorted(outcomes.items(), key=lambda item: -item[1])
                ]
            ),
            hide_index=True,
            column_config={"Share": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)},
            alt="Known labels of the events in this alert",
        )
        st.caption("Real labels are only known because this is a labelled dataset. A real network wouldn't have them.")

with st.expander("Network details"):
    st.dataframe(
        pd.DataFrame(
            [
                {"Field": "Source address", "Value": alert["src_ip"] or "Not available"},
                {"Field": "Destination address", "Value": alert["dst_ip"] or "Not available"},
                {"Field": "Protocol", "Value": alert["protocol"] or "Not available"},
                {"Field": "First seen (UTC)", "Value": alert["created_at"][:19].replace("T", " ")},
                {"Field": "Last seen (UTC)", "Value": alert["last_seen_at"][:19].replace("T", " ")},
            ]
        ),
        hide_index=True,
        alt="Network details for this alert",
    )
    if not alert["src_ip"]:
        st.caption("The dataset has no addresses, so these are left empty.")
