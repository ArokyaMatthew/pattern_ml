"""Matplotlib figures for the workbench.

Every function returns a ``Figure`` and leaves showing or saving it to the
caller. Styling follows a few fixed rules: thin marks, hairline solid
gridlines, text in ink tones (never in a series colour), a legend whenever
two or more series share a chart, and selective direct labels.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from arbor.metrics import METRIC_LABELS, METRICS  # noqa: E402
from arbor.style import (  # noqa: E402
    AXIS,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    PAGE,
    SEQUENTIAL,
    algorithm_color,
    class_color,
    mix,
    text_on,
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 8.5,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.facecolor": PAGE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 9.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.facecolor": PAGE,
        "figure.dpi": 100,
        "savefig.facecolor": PAGE,
        "savefig.dpi": 200,
    }
)

_SEQUENTIAL_MAP = LinearSegmentedColormap.from_list("arbor_blue", SEQUENTIAL)


def _short(label, width: int = 16) -> str:
    text = str(label)
    return text if len(text) <= width else text[: width - 1] + "…"


def _value_grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def class_distribution(counts: pd.Series, classes: list) -> Figure:
    """Horizontal bars of class frequencies, each bar in its class colour."""
    counts = counts.reindex(classes).fillna(0)
    total = counts.sum()
    n = len(counts)
    fig, ax = plt.subplots(figsize=(3.9, 0.34 * n + 0.45))
    y = np.arange(n)[::-1]
    colors = [class_color(i) for i in range(n)]
    ax.barh(y, counts.to_numpy(), height=0.56, color=colors, zorder=2)
    ax.set_yticks(y, [_short(c, 16) for c in counts.index])
    for yi, value in zip(y, counts.to_numpy()):
        ax.text(value, yi, f"  {value:,.0f} · {value / total:.0%}", va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlim(0, counts.max() * 1.42)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    fig.tight_layout(pad=0.4)
    return fig


def confusion_matrix(cm: np.ndarray, labels: list, title: str | None = None) -> Figure:
    """Counts with cells shaded by the share of each actual class (row %)."""
    n = len(labels)
    side = min(5.6, 1.5 + 0.44 * n)
    fig, ax = plt.subplots(figsize=(side + 0.9, side + 0.35))
    totals = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, totals, out=np.zeros(cm.shape, dtype=float), where=totals > 0)
    ax.imshow(share, cmap=_SEQUENTIAL_MAP, vmin=0, vmax=1, aspect="auto")

    count_size = 10 if n <= 6 else 8 if n <= 10 else 6.5
    for i in range(n):
        for j in range(n):
            fill = _SEQUENTIAL_MAP(share[i, j])
            color = text_on(to_hex(fill))
            ax.text(j, i - (0.09 if n <= 8 else 0), f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=count_size, fontweight="semibold", color=color)
            if n <= 8:
                ax.text(j, i + 0.2, f"{share[i, j]:.0%}", ha="center", va="center", fontsize=7.5, color=color)

    # Surface-coloured gaps between cells instead of drawn borders.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=PAGE, linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    short = [_short(label, 14) for label in labels]
    ax.set_xticks(range(n), short, rotation=30 if max(map(len, short)) > 6 and n > 3 else 0,
                  ha="right" if max(map(len, short)) > 6 and n > 3 else "center")
    ax.set_yticks(range(n), short)
    ax.set_xlabel("Predicted class", labelpad=8)
    ax.set_ylabel("Actual class", labelpad=8)
    if title:
        ax.set_title(title)
    fig.tight_layout(pad=0.5)
    return fig


def metric_comparison(scores: pd.DataFrame) -> Figure:
    """Grouped columns: one group per metric, one column per algorithm."""
    algorithms = list(scores.index)
    k = len(algorithms)
    fig, ax = plt.subplots(figsize=(8.6, 2.5))
    x = np.arange(len(METRICS))
    width = 0.15 if k <= 2 else 0.12
    step = width * 1.06  # small surface gap between neighbouring columns
    for i, algorithm in enumerate(algorithms):
        offset = (i - (k - 1) / 2) * step
        values = scores.loc[algorithm, list(METRICS)].to_numpy(dtype=float)
        ax.bar(x + offset, values, width=width, color=algorithm_color(algorithm), label=algorithm, zorder=2)
        for xi, value in zip(x + offset, values):
            ax.text(xi, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)
    ax.set_xticks(x, [METRIC_LABELS[m] for m in METRICS])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    _value_grid(ax)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.2), ncol=k, handlelength=1, handleheight=1, columnspacing=1.4)
    fig.tight_layout(pad=0.4)
    return fig


def cv_scores(cv: pd.DataFrame, metric: str = "accuracy") -> Figure:
    """Dot plot of per-fold scores with the mean marked for each algorithm."""
    algorithms = list(dict.fromkeys(cv["algorithm"]))
    n = len(algorithms)
    fig, ax = plt.subplots(figsize=(8.6, 0.6 * n + 0.85))
    low = max(0.0, float(cv[metric].min()) - 0.05)
    for row, algorithm in enumerate(algorithms):
        y = n - 1 - row
        values = cv.loc[cv["algorithm"] == algorithm, metric].to_numpy(dtype=float)
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(values, y + jitter, s=46, color=algorithm_color(algorithm), edgecolor=PAGE,
                   linewidth=1.5, zorder=3, label=algorithm)
        mean, std = values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0
        ax.plot([mean, mean], [y - 0.28, y + 0.28], color=INK, linewidth=2, zorder=4, solid_capstyle="round")
        ax.text(1.005, y, f"mean {mean:.3f}  ± {std:.3f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_yticks(range(n), algorithms[::-1])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(low - 0.02, 1.02)
    ax.set_xlabel(f"{METRIC_LABELS[metric]} per fold", labelpad=6)
    _value_grid(ax, axis="x")
    ax.spines["left"].set_visible(False)
    fig.tight_layout(pad=0.4, rect=(0, 0, 0.84, 1))
    return fig


def depth_curves(sweep: pd.DataFrame) -> Figure:
    """Small multiples of train vs test accuracy against the depth limit."""
    algorithms = list(dict.fromkeys(sweep["algorithm"]))
    fig, axes = plt.subplots(1, len(algorithms), figsize=(8.6, 2.8), sharey=True, squeeze=False)
    low = max(0.0, min(sweep["train"].min(), sweep["test"].min()) - 0.05)
    for ax, algorithm in zip(axes[0], algorithms):
        data = sweep[sweep["algorithm"] == algorithm]
        color = algorithm_color(algorithm)
        light = mix(color, PAGE, 0.55)
        depth = data["depth"].to_numpy()
        for column, tone, label in (("train", light, "train"), ("test", color, "test")):
            values = data[column].to_numpy()
            ax.plot(depth, values, color=tone, linewidth=2, solid_joinstyle="round", solid_capstyle="round", zorder=2)
            ax.scatter(depth[-1:], values[-1:], s=36, color=tone, edgecolor=PAGE, linewidth=1.5, zorder=3)
            ax.annotate(label, (depth[-1], values[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK_SECONDARY)
        best = data.loc[data["test"].idxmax()]
        ax.scatter([best["depth"]], [best["test"]], s=70, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=4)
        ax.set_title(algorithm)
        ax.text(1.0, 1.0, f"best test {best['test']:.3f} at depth {int(best['depth'])} (ringed)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK_SECONDARY)
        ax.set_xlabel("Maximum depth", labelpad=6)
        ax.set_xticks(depth if len(depth) <= 12 else depth[:: max(1, len(depth) // 8)])
        ax.set_xlim(depth.min() - 0.3, depth.max() + 1.2)
        _value_grid(ax)
        ax.spines["left"].set_visible(False)
    axes[0][0].set_ylabel("Accuracy", labelpad=6)
    axes[0][0].set_ylim(low, 1.03)
    fig.tight_layout(pad=0.4, w_pad=2.5)
    return fig


def feature_importance(importance: pd.DataFrame, top: int = 10) -> Figure:
    """Grouped horizontal bars; ``importance`` has features as rows, algorithms as columns."""
    data = importance.loc[importance.max(axis=1).sort_values(ascending=False).index[:top]]
    data = data[(data > 0).any(axis=1)] if (data > 0).any().any() else data
    n, k = len(data), data.shape[1]
    fig, ax = plt.subplots(figsize=(4.3, 0.36 * n * max(1, k / 2) + 0.7))
    height = 0.8 / k
    y = np.arange(n)[::-1]
    for i, algorithm in enumerate(data.columns):
        offset = (i - (k - 1) / 2) * height * 1.08
        ax.barh(y - offset, data[algorithm].to_numpy(), height=height, color=algorithm_color(algorithm),
                label=algorithm, zorder=2)
    ax.set_yticks(y, [_short(f, 18) for f in data.index])
    ax.set_xlim(0, max(0.05, float(data.to_numpy().max())) * 1.1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _value_grid(ax, axis="x")
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_visible(False)
    ax.legend(loc="lower right", handlelength=1, handleheight=1)
    fig.tight_layout(pad=0.4)
    return fig
