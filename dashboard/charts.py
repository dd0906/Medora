"""Altair charts for the dashboard, with colors tied to classes and severities."""

import altair as alt
import pandas as pd

from dashboard.views import class_label
from detection.config import CLASSES

# Fixed per class, so a class keeps its color when another has no data.
CLASS_COLORS = dict(zip((class_label(name) for name in CLASSES), ("#3987e5", "#d95926", "#199e70", "#c98500")))
SEVERITY_COLORS = {"Critical": "#F87171", "High": "#FB923C", "Medium": "#FBBF24", "Low": "#60A5FA"}
HEATMAP_RANGE = ["#1E293B", "#1c5cab", "#86b6ef"]


def _class_scale() -> alt.Scale:
    return alt.Scale(domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values()))


def _room_for_labels(values: pd.Series) -> alt.Scale:
    """Extend the axis past the longest bar, so its number label isn't cut off."""
    longest = float(values.max()) if len(values) else 0.0
    return alt.Scale(domain=[0, max(longest * 1.18, 1)], nice=False)


def class_bars(frame: pd.DataFrame) -> alt.Chart:
    """Event count per class, labelled directly on each bar."""
    base = alt.Chart(frame).encode(
        y=alt.Y("class:N", sort=list(CLASS_COLORS), title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X(
            "events:Q",
            title="Records",
            scale=_room_for_labels(frame["events"]),
            axis=alt.Axis(format="~s", tickMinStep=1, grid=False, tickCount=5),
        ),
        tooltip=[alt.Tooltip("class:N", title="Type"), alt.Tooltip("events:Q", title="Records", format=",")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, size=20).encode(color=alt.Color("class:N", scale=_class_scale(), legend=None))
    labels = base.mark_text(align="left", dx=4, color="#F1F5F9").encode(text=alt.Text("events:Q", format=","))
    return (bars + labels).properties(height=190)


def severity_bars(frame: pd.DataFrame) -> alt.Chart:
    """Alert count per severity, with severity also named on the axis."""
    base = alt.Chart(frame).encode(
        y=alt.Y("severity:N", sort=list(SEVERITY_COLORS), title=None),
        x=alt.X(
            "alerts:Q",
            title="Alerts",
            scale=_room_for_labels(frame["alerts"]),
            axis=alt.Axis(tickMinStep=1, format="d", grid=False),
        ),
        tooltip=[alt.Tooltip("severity:N", title="Severity"), alt.Tooltip("alerts:Q", title="Alerts", format=",")],
    )
    color = alt.Color(
        "severity:N",
        scale=alt.Scale(domain=list(SEVERITY_COLORS), range=list(SEVERITY_COLORS.values())),
        legend=None,
    )
    bars = base.mark_bar(cornerRadiusEnd=4, size=20).encode(color=color)
    labels = base.mark_text(align="left", dx=4, color="#F1F5F9").encode(text="alerts:Q")
    return (bars + labels).properties(height=190)


def timeline_area(frame: pd.DataFrame) -> alt.Chart:
    """Stacked event counts over replay time."""
    return (
        alt.Chart(frame)
        .mark_area()
        .encode(
            # A UTC scale, or the browser would show its own time zone under a UTC title.
            x=alt.X(
                "time:T",
                title="Processing time (UTC)",
                scale=alt.Scale(type="utc"),
                axis=alt.Axis(format="%H:%M:%S", labelOverlap=True),
            ),
            y=alt.Y("events:Q", stack="zero", title="Records per interval", axis=alt.Axis(format="~s")),
            color=alt.Color(
                "class:N",
                scale=_class_scale(),
                legend=alt.Legend(title=None, orient="top", labelLimit=220),
            ),
            tooltip=[
                alt.Tooltip("time:T", title="Time (UTC)", format="%H:%M:%S", formatType="utc"),
                alt.Tooltip("class:N", title="Type"),
                alt.Tooltip("events:Q", title="Records", format=","),
            ],
        )
        .properties(height=300)
    )


def effect_bars(frame: pd.DataFrame) -> alt.Chart:
    """How much each feature pushed the model towards its answer."""
    base = alt.Chart(frame).encode(
        y=alt.Y("Feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X(
            "Push:Q",
            title="Push towards this answer (percentage points)",
            scale=_room_for_labels(frame["Push"]),
            axis=alt.Axis(grid=False, tickCount=5),
        ),
        tooltip=[
            alt.Tooltip("Feature:N"),
            alt.Tooltip("Meaning:N"),
            alt.Tooltip("Push:Q", title="Push (points)", format=".1f"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, size=18, color="#5B9BE6")
    labels = base.mark_text(align="left", dx=4, color="#F1F5F9").encode(text=alt.Text("Push:Q", format="+.1f"))
    return (bars + labels).properties(height=alt.Step(30))


def confusion_heatmap(frame: pd.DataFrame) -> alt.Chart:
    """Actual against predicted class, shaded by share of the actual class."""
    order = list(CLASS_COLORS)
    data = frame.copy()
    totals = data.groupby("Actual")["Records"].transform("sum")
    data["Share"] = (data["Records"] / totals.where(totals > 0)).fillna(0)
    base = alt.Chart(data).encode(
        x=alt.X("Predicted:N", sort=order, axis=alt.Axis(labelAngle=-30, labelLimit=170, labelOverlap=False)),
        y=alt.Y("Actual:N", sort=order, axis=alt.Axis(labelLimit=170, labelOverlap=False)),
        tooltip=[
            "Actual:N",
            "Predicted:N",
            alt.Tooltip("Records:Q", format=","),
            alt.Tooltip("Share:Q", title="Share of actual class", format=".1%"),
        ],
    )
    cells = base.mark_rect(stroke="#0F172A", strokeWidth=2).encode(
        color=alt.Color(
            "Share:Q",
            scale=alt.Scale(domain=[0, 0.5, 1], range=HEATMAP_RANGE),
            legend=alt.Legend(title="Share of actual", format=".0%", orient="right"),
        )
    )
    labels = base.mark_text(fontSize=13).encode(
        text=alt.Text("Records:Q", format=","),
        color=alt.condition(alt.datum.Share > 0.7, alt.value("#0F172A"), alt.value("#F1F5F9")),
    )
    return (cells + labels).properties(height=340)
