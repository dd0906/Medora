import pandas as pd
import streamlit as st

from dashboard import charts, views
from dashboard.api_client import ApiError, ApiUnavailable
from dashboard.session import get_client, show_api_problem
from detection.config import CLASSES

client = get_client()

try:
    model = client.model()
    evaluation = client.evaluation()
except (ApiUnavailable, ApiError) as error:
    show_api_problem(error)

st.caption(f"Model in use: {model['version']} · {model['algorithm']} · {len(model['feature_columns'])} features")

if not evaluation["available"]:
    st.warning(evaluation["reason"], title="No saved evaluation for this model", icon=":material/warning:")
    st.stop()

metrics = evaluation["test_metrics"]
comparison = evaluation["comparison"]
arp = metrics["per_class"]["ARP_Spoofing"]

st.warning(
    "The test data seems to come from the same recordings as the training data, so these scores may not hold "
    "on other networks.",
    title="Keep in mind",
    icon=":material/warning:",
)

with st.container(horizontal=True):
    st.metric(
        "Macro F1", f"{metrics['macro_f1']:.2%}", border=True,
        help="The main score. It averages all four types equally, so a weak type can't hide behind common ones.",
    )
    st.metric(
        "Accuracy", f"{metrics['accuracy']:.2%}", border=True,
        help="Share of all records labelled correctly. It looks high even when a rare type is often missed.",
    )
    st.metric("ARP F1", f"{arp['f1']:.2%}", border=True, help="Score for ARP spoofing, the hardest type to catch.")
    st.metric(
        "False alarms", f"{metrics['benign_false_alarm_rate']:.2%}", border=True,
        help="Share of normal records wrongly flagged as attacks.",
    )
st.write(views.performance_summary(metrics))

matrix_col, effects_col = st.columns(2)
with matrix_col.container(border=True):
    st.subheader("Right and wrong predictions")
    st.altair_chart(
        charts.confusion_heatmap(views.confusion_matrix_long(metrics["confusion_matrix"])),
        alt="Heatmap of actual class against predicted class on the supplied test split",
    )
    st.caption("Rows are the real type, and columns are what the model predicted. The diagonal is where it was right.")

effects = evaluation.get("feature_effects")
with effects_col.container(border=True):
    st.subheader("What the model looks at")
    if effects and effects["types"]:
        available = [name for name in CLASSES if name in effects["types"]]
        chosen = st.segmented_control(
            "Attack type", options=available, format_func=views.class_label, default=available[-1], key="effects_type"
        ) or available[-1]
        summary = effects["types"][chosen]
        st.altair_chart(
            charts.effect_bars(views.effects_frame(summary["features"], "average_effect")),
            alt="Features that pushed the model most towards this type on average",
        )
        st.caption(
            f"Averaged over {summary['records']:,} test records the model labelled as {views.class_label(chosen)}. "
            "This shows what the model relies on, not what causes the attack."
        )
    else:
        st.write("Run `python -m scripts.explain_model` to see which features the model relies on.")

st.subheader("More detail")
st.caption("Background checks behind these results. The full write-up is in docs/results.md.")

with st.expander("Scores for each attack type"):
    st.dataframe(
        views.per_class_table(metrics),
        hide_index=True,
        column_config={
            name: st.column_config.NumberColumn(format="%.3f") for name in ("Precision", "Recall", "F1")
        } | {"Records": st.column_config.NumberColumn(format="localized")},
        alt="Precision, recall, and F1 for each class",
    )
    st.caption(
        "Precision: how often an alert of this type was right. Recall: how many real cases were caught. "
        f"ARP spoofing is the weakest: {1 - arp['precision']:.0%} of its predictions were wrong, "
        f"and {1 - arp['recall']:.0%} of real cases were missed."
    )

with st.expander("Models compared"):
    st.dataframe(
        views.comparison_table(comparison, evaluation["selected"]),
        hide_index=True,
        column_config={
            "In use": st.column_config.CheckboxColumn(),
            "False alarms": st.column_config.NumberColumn(format="percent"),
            **{
                name: st.column_config.NumberColumn(format="%.4f")
                for name in (
                    "Macro F1 (training check)", "Macro F1 (test)", "Accuracy (test)",
                    "ARP F1 (training check)", "ARP F1 (test)",
                )
            },
        },
        alt="Cross-validation and test scores for every candidate model",
    )
    st.caption("The model was chosen using training data only, before any test results were seen.")
    st.caption(
        "\"rf with iat\" is only for comparison. It scores higher partly because one of its features seems to "
        "recognise the recording rather than the attack."
    )

attempt = evaluation["arp_attempt"]
if attempt:
    with st.expander("Trying to improve ARP detection"):
        active = next(row for row in comparison if row["model"] == evaluation["selected"])
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Model": "Model in use",
                        "ARP F1": active["arp_f1"],
                        "ARP precision": active["arp_precision"],
                        "ARP recall": active["arp_recall"],
                        "Macro F1": active["macro_f1"],
                        "False alarms": active["benign_false_alarm_rate"],
                    },
                    {
                        "Model": f"Not used: {attempt['candidate']}",
                        "ARP F1": attempt["arp_f1"],
                        "ARP precision": attempt["arp_precision"],
                        "ARP recall": attempt["arp_recall"],
                        "Macro F1": attempt["macro_f1"],
                        "False alarms": attempt["benign_false_alarm_rate"],
                    },
                ]
            ),
            hide_index=True,
            column_config={
                "False alarms": st.column_config.NumberColumn(format="percent"),
                **{
                    name: st.column_config.NumberColumn(format="%.4f")
                    for name in ("ARP F1", "ARP precision", "ARP recall", "Macro F1")
                },
            },
            alt="Active model compared with the rejected ARP candidate on the test split",
        )
        st.caption("This version looked better on training data but did worse on test data, so it isn't used.")

arp_rules = evaluation["arp_rules"]
if arp_rules:
    with st.expander("ARP rules on capture files"):
        attacks = [row for row in arp_rules if row["label"] == "arp_spoofing"]
        benign = [row for row in arp_rules if row["label"] == "benign"]
        st.write(
            f"The rules caught {sum(row['flagged'] for row in attacks)} of {len(attacks)} attack captures, but also "
            f"flagged {sum(row['flagged'] for row in benign)} of {len(benign)} normal ones. Treat their alerts as leads "
            "to check."
        )
        st.dataframe(
            views.arp_rule_table(arp_rules),
            hide_index=True,
            column_config={
                "Flagged": st.column_config.CheckboxColumn(),
                "Hours": st.column_config.NumberColumn(format="%.1f"),
                "First finding after (s)": st.column_config.NumberColumn(format="%.1f"),
            },
            alt="ARP rule results for each labelled capture",
        )
        st.caption(
            "Results are per capture file, so they can't be compared with the model's scores. All captures seem to "
            "come from one lab."
        )

device_baselines = evaluation["device_baselines"]
if device_baselines:
    with st.expander("Device baselines"):
        st.write(
            "Each device is compared with its own normal behaviour, with no attack examples needed. These captures "
            "weren't used for learning or tuning."
        )
        st.dataframe(
            views.device_baseline_table(device_baselines),
            hide_index=True,
            column_config={
                "Hours": st.column_config.NumberColumn(format="%.2f"),
                "Findings per hour": st.column_config.NumberColumn(format="%.1f"),
                "Largest rise": st.column_config.NumberColumn(
                    format="%.0f times",
                    help="How far the most unusual device went above its own normal limit.",
                ),
            },
            alt="Device baseline findings for each held-out capture",
        )
        st.caption(
            "Port scans and floods stand out clearly. ARP spoofing doesn't. The attack captures are short, and all "
            "come from one lab."
        )

outside = evaluation["outside_data"]
if outside:
    with st.expander("Test on another dataset (CICIoT2023)"):
        st.write(
            "The model can't run on CICIoT2023 directly, because its features are worked out differently. These quick "
            "test models use only the columns both datasets share."
        )
        st.dataframe(
            pd.DataFrame(outside["checks"]).rename(
                columns={
                    "model": "Fitted on",
                    "feature_count": "Features",
                    "scored_on": "Scored on",
                    "accuracy": "Accuracy",
                    "macro_f1": "Macro F1",
                    "attack_detection_rate": "Attacks flagged",
                    "benign_false_alarm_rate": "False alarms",
                    "arp_recall": "ARP recall",
                }
            ),
            hide_index=True,
            column_config={
                "Accuracy": st.column_config.NumberColumn(format="percent"),
                "Macro F1": st.column_config.NumberColumn(format="%.4f"),
                "Attacks flagged": st.column_config.NumberColumn(format="percent"),
                "False alarms": st.column_config.NumberColumn(format="percent"),
                "ARP recall": st.column_config.NumberColumn(format="percent"),
            },
            alt="Throwaway model scores on CICIoMT2024 and CICIoT2023",
        )
        st.caption(
            f"Simply guessing \"normal\" every time would score {outside['always_benign_accuracy']:.2%} here, so "
            "these models don't carry over to this dataset. A fair test would need both datasets processed the same way."
        )
