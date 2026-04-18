from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

try:
    from .algorithm import SinkhornResult, SinkhornTraceFrame, EPS
except ImportError:
    from algorithm import SinkhornResult, SinkhornTraceFrame, EPS


def _default_labels(prefix: str, size: int) -> list[str]:
    return [f"{prefix}{index}" for index in range(size)]


def plot_matrix_heatmap(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    cmap: str = "cividis",
    value_fmt: str = "{:.3f}",
    show_colorbar: bool = False,
    annotation_fontsize: int = 10,
) -> plt.Axes:
    values = np.asarray(matrix, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(values, cmap=cmap, aspect="auto")
    if show_colorbar:
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    n_rows, n_cols = values.shape
    row_labels = list(row_labels or _default_labels("row_", n_rows))
    col_labels = list(col_labels or _default_labels("col_", n_cols))

    ax.set_xticks(np.arange(n_cols), labels=col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_rows), labels=row_labels)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    norm = image.norm
    cmap_obj = image.cmap
    for row in range(n_rows):
        for col in range(n_cols):
            rgba = cmap_obj(norm(values[row, col]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if luminance < 0.45 else "#111111"
            outline_color = "#111111" if text_color == "white" else "white"
            ax.text(
                col,
                row,
                value_fmt.format(values[row, col]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=annotation_fontsize,
                fontweight="semibold",
                bbox={
                    "facecolor": (1, 1, 1, 0.12) if text_color == "white" else (1, 1, 1, 0.35),
                    "edgecolor": "none",
                    "pad": 0.15,
                },
                path_effects=[patheffects.withStroke(linewidth=1.3, foreground=outline_color, alpha=0.7)],
            )

    if title is not None:
        ax.set_title(title)
    return ax


def plot_distribution(
    weights: Sequence[float] | np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    positions: Sequence[float] | np.ndarray | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    color: str = "#457b9d",
) -> plt.Axes:
    values = np.asarray(weights, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    labels = list(labels or _default_labels("mass_", values.shape[0]))
    x = np.arange(values.shape[0]) if positions is None else np.asarray(positions, dtype=float)
    bar_width = 0.8 if positions is None else 0.18 * max(1.0, (x.max() - x.min()) / max(1, values.shape[0] - 1))
    ax.bar(x, values, width=bar_width, color=color, alpha=0.85)

    if positions is None:
        ax.set_xticks(x, labels=labels, rotation=40, ha="right")
    else:
        ax.set_xticks(x, labels=labels, rotation=40, ha="right")

    ax.set_ylim(0.0, max(0.45, float(values.max()) * 1.25))
    ax.set_ylabel("mass")
    ax.grid(alpha=0.25, axis="y")
    if title is not None:
        ax.set_title(title)
    return ax


def plot_marginal_comparison(
    target: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    target_label: str = "target",
    current_label: str = "current",
) -> plt.Axes:
    target_values = np.asarray(target, dtype=float)
    current_values = np.asarray(current, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(target_values.shape[0])
    labels = list(labels or _default_labels("mass_", target_values.shape[0]))
    ax.bar(x - 0.18, target_values, width=0.34, color="#457b9d", label=target_label, alpha=0.85)
    ax.bar(x + 0.18, current_values, width=0.34, color="#e76f51", label=current_label, alpha=0.75)
    ax.set_xticks(x, labels=labels, rotation=40, ha="right")
    ax.set_ylim(0.0, max(0.45, float(max(target_values.max(), current_values.max())) * 1.25))
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right")
    if title is not None:
        ax.set_title(title)
    return ax


def plot_transport_graph(
    source_positions: Sequence[Sequence[float]] | np.ndarray,
    target_positions: Sequence[Sequence[float]] | np.ndarray,
    source_weights: Sequence[float] | np.ndarray,
    target_weights: Sequence[float] | np.ndarray,
    coupling: Sequence[Sequence[float]] | np.ndarray,
    *,
    source_labels: Sequence[str] | None = None,
    target_labels: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    source = np.asarray(source_positions, dtype=float)
    target = np.asarray(target_positions, dtype=float)
    coupling = np.asarray(coupling, dtype=float)
    source_weights = np.asarray(source_weights, dtype=float)
    target_weights = np.asarray(target_weights, dtype=float)

    if source.ndim == 1:
        source_xy = np.column_stack([source, np.ones_like(source)])
    else:
        source_xy = source

    if target.ndim == 1:
        target_xy = np.column_stack([target, np.zeros_like(target)])
    else:
        target_xy = target

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    source_labels = list(source_labels or _default_labels("source_", source_xy.shape[0]))
    target_labels = list(target_labels or _default_labels("target_", target_xy.shape[0]))

    max_mass = float(coupling.max()) if coupling.size else 1.0
    cutoff = max_mass * 0.03
    for row in range(coupling.shape[0]):
        for col in range(coupling.shape[1]):
            mass = coupling[row, col]
            if mass < cutoff:
                continue
            width = 0.5 + 5.0 * mass / max_mass
            alpha = 0.20 + 0.65 * mass / max_mass
            ax.plot(
                [source_xy[row, 0], target_xy[col, 0]],
                [source_xy[row, 1], target_xy[col, 1]],
                color="#1d3557",
                linewidth=width,
                alpha=alpha,
                zorder=1,
            )

    source_sizes = 1400 * source_weights / max(float(source_weights.max()), EPS)
    target_sizes = 1400 * target_weights / max(float(target_weights.max()), EPS)
    ax.scatter(source_xy[:, 0], source_xy[:, 1], s=source_sizes, color="#457b9d", zorder=3, label="source")
    ax.scatter(
        target_xy[:, 0],
        target_xy[:, 1],
        s=target_sizes,
        color="#e76f51",
        zorder=3,
        marker="s",
        label="target",
    )

    for row, label in enumerate(source_labels):
        ax.text(source_xy[row, 0] + 0.03, source_xy[row, 1] + 0.03, label, color="#1d3557")
    for col, label in enumerate(target_labels):
        ax.text(target_xy[col, 0] + 0.03, target_xy[col, 1] + 0.03, label, color="#9c2c13")

    ax.grid(alpha=0.22)
    ax.legend(loc="upper right")
    if source.ndim == 1 and target.ndim == 1:
        ax.set_yticks([0.0, 1.0], labels=["target line", "source line"])
    else:
        ax.set_aspect("equal", adjustable="box")
    if title is not None:
        ax.set_title(title)
    return ax


def plot_convergence(result: SinkhornResult, *, ax: plt.Axes | None = None, title: str | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    iterations = np.arange(1, result.row_l1_history.shape[0] + 1)
    ax.plot(iterations, result.row_l1_history, marker="o", color="#457b9d", label="row L1 error")
    ax.plot(iterations, result.col_l1_history, marker="s", color="#e76f51", label="column L1 error")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("L1 marginal error")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    if title is not None:
        ax.set_title(title)
    return ax


def plot_trace_frame(
    frame: SinkhornTraceFrame,
    source_weights: Sequence[float] | np.ndarray,
    target_weights: Sequence[float] | np.ndarray,
    *,
    source_labels: Sequence[str] | None = None,
    target_labels: Sequence[str] | None = None,
    figure: plt.Figure | None = None,
) -> plt.Figure:
    if figure is None:
        figure = plt.figure(figsize=(13.5, 4.8))

    figure.clf()
    axes = figure.subplots(1, 3)

    plot_matrix_heatmap(
        frame.coupling,
        row_labels=source_labels,
        col_labels=target_labels,
        ax=axes[0],
        title=f"Coupling at iteration {frame.iteration}",
        cmap="magma",
        value_fmt="{:.3f}",
        show_colorbar=False,
    )
    plot_marginal_comparison(
        source_weights,
        frame.row_sum,
        labels=source_labels,
        ax=axes[1],
        title="Row Marginals",
        target_label="source a",
        current_label="current rows",
    )
    plot_marginal_comparison(
        target_weights,
        frame.col_sum,
        labels=target_labels,
        ax=axes[2],
        title="Column Marginals",
        target_label="target b",
        current_label="current cols",
    )

    figure.suptitle(f"{frame.stage}: {frame.note}", fontsize=11, y=1.02)
    figure.tight_layout()
    return figure


__all__ = [
    "plot_convergence",
    "plot_distribution",
    "plot_marginal_comparison",
    "plot_matrix_heatmap",
    "plot_trace_frame",
    "plot_transport_graph",
]
