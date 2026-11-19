"""Static report figures for the evaluation protocol."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SERIES_CV = "#2a78d6"
SERIES_TEST = "#eb6834"


def _style(axis: plt.Axes) -> None:
    axis.set_facecolor(SURFACE)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#d9d8d4")
    axis.tick_params(colors=TEXT_SECONDARY, length=0)


def plot_comparison(comparison: pd.DataFrame, path: Path) -> None:
    """Paired CV and test bars per model, for overall and ARP performance."""
    ordered = comparison.sort_values("macro_f1")
    positions = np.arange(len(ordered))
    panels = [
        ("Macro F1", "cv_macro_f1", "macro_f1"),
        ("ARP Spoofing F1", "cv_arp_f1", "arp_f1"),
    ]

    figure, axes = plt.subplots(
        1, 2, figsize=(11, 0.75 * len(ordered) + 1.8), sharey=True, facecolor=SURFACE
    )
    height = 0.34
    for axis, (title, cv_column, test_column) in zip(axes, panels):
        cv_bars = axis.barh(positions + height / 2 + 0.02, ordered[cv_column], height, color=SERIES_CV, label="Cross-validation on train")
        test_bars = axis.barh(positions - height / 2 - 0.02, ordered[test_column], height, color=SERIES_TEST, label="Supplied test split")
        for bars in (cv_bars, test_bars):
            axis.bar_label(bars, fmt="%.3f", padding=3, color=TEXT_PRIMARY, fontsize=8)
        axis.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=11)
        axis.set_xlim(0, 1.1)
        axis.set_yticks(positions, ordered["model"])
        _style(axis)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", frameon=False, ncols=2, labelcolor=TEXT_SECONDARY, fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_confusion(matrix: np.ndarray, classes: Sequence[str], model_name: str, path: Path) -> None:
    """Confusion matrix shaded by recall per row and labelled with raw counts."""
    rates = matrix / matrix.sum(axis=1, keepdims=True)
    colormap = LinearSegmentedColormap.from_list("medora_blue", ["#f4f8fd", "#86b6ef", "#184f95"])

    figure, axis = plt.subplots(figsize=(7.5, 5.8), facecolor=SURFACE)
    axis.imshow(rates, cmap=colormap, vmin=0, vmax=1)
    labels = [name.replace("_", " ") for name in classes]
    axis.set_xticks(range(len(classes)), labels, rotation=20, ha="right", color=TEXT_SECONDARY, fontsize=9)
    axis.set_yticks(range(len(classes)), labels, color=TEXT_SECONDARY, fontsize=9)
    axis.set_xlabel("Predicted", color=TEXT_SECONDARY)
    axis.set_ylabel("Actual", color=TEXT_SECONDARY)
    axis.set_title(f"{model_name}: supplied test split (shade = share of actual class)", loc="left", color=TEXT_PRIMARY, fontsize=10)
    axis.tick_params(length=0)
    axis.spines[:].set_visible(False)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "#ffffff" if rates[row, column] > 0.55 else TEXT_PRIMARY
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", color=color, fontsize=10)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
