from __future__ import annotations

import base64
import tempfile
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from matplotlib import patheffects

try:
    from .algorithm import HungarianResult, TraceFrame
except ImportError:
    from algorithm import HungarianResult, TraceFrame


def _default_labels(prefix: str, size: int) -> list[str]:
    return [f"{prefix}{index}" for index in range(size)]


def _extend_labels(labels: Sequence[str] | None, size: int, prefix: str) -> list[str]:
    labels = list(labels or _default_labels(prefix, size))
    if len(labels) < size:
        labels.extend(f"{prefix}{index}" for index in range(len(labels), size))
    return labels[:size]


def plot_cost_heatmap(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    assignment: Sequence[tuple[int, int]] | None = None,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    cmap: str = "YlGnBu",
    value_fmt: str = "{:.2f}",
    highlight_zeros: bool = False,
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
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground=outline_color, alpha=0.65)],
            )

    if assignment is not None:
        for row, col in assignment:
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    linewidth=2.5,
                    edgecolor="#d62828",
                )
            )

    if highlight_zeros:
        zero_rows, zero_cols = np.where(np.isclose(values, 0.0, atol=1e-9))
        ax.scatter(
            zero_cols,
            zero_rows,
            s=220,
            facecolors="none",
            edgecolors="#2a9d8f",
            linewidths=2,
        )

    if title is not None:
        ax.set_title(title)
    return ax


def plot_bipartite_assignment(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    assignment: Sequence[tuple[int, int]],
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_all_edges: bool | None = None,
) -> plt.Axes:
    values = np.asarray(matrix, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    n_rows, n_cols = values.shape
    if show_all_edges is None:
        show_all_edges = values.size <= 49

    row_labels = list(row_labels or _default_labels("row_", n_rows))
    col_labels = list(col_labels or _default_labels("col_", n_cols))
    left_y = np.linspace(1.0, 0.0, n_rows)
    right_y = np.linspace(1.0, 0.0, n_cols)

    max_cost = float(values.max()) if values.size else 1.0
    if show_all_edges:
        for row in range(n_rows):
            for col in range(n_cols):
                normalized = values[row, col] / max_cost if max_cost else 0.0
                alpha = 0.10 + 0.25 * (1.0 - normalized)
                width = 0.5 + 1.0 * (1.0 - normalized)
                ax.plot([0.0, 1.0], [left_y[row], right_y[col]], color="0.55", alpha=alpha, linewidth=width, zorder=1)

    for row, col in assignment:
        ax.plot([0.0, 1.0], [left_y[row], right_y[col]], color="#1d3557", linewidth=3.0, zorder=3)
        ax.text(
            0.5,
            (left_y[row] + right_y[col]) / 2.0,
            f"{values[row, col]:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color="#1d3557",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
            zorder=4,
        )

    ax.scatter(np.zeros(n_rows), left_y, s=300, color="#457b9d", zorder=5)
    ax.scatter(np.ones(n_cols), right_y, s=300, color="#e76f51", zorder=5)

    for row, label in enumerate(row_labels):
        ax.text(-0.04, left_y[row], label, ha="right", va="center")
    for col, label in enumerate(col_labels):
        ax.text(1.04, right_y[col], label, ha="left", va="center")

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    return ax


def plot_spatial_assignment(
    row_positions: np.ndarray,
    col_positions: np.ndarray,
    assignment: Sequence[tuple[int, int]],
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    row_labels = list(row_labels or _default_labels("row_", row_positions.shape[0]))
    col_labels = list(col_labels or _default_labels("col_", col_positions.shape[0]))

    ax.scatter(row_positions[:, 0], row_positions[:, 1], s=250, color="#457b9d", label="workers", zorder=3)
    ax.scatter(col_positions[:, 0], col_positions[:, 1], s=250, color="#e76f51", marker="s", label="jobs", zorder=3)

    for row, label in enumerate(row_labels):
        ax.text(row_positions[row, 0] + 0.03, row_positions[row, 1] + 0.03, label, color="#1d3557")
    for col, label in enumerate(col_labels):
        ax.text(col_positions[col, 0] + 0.03, col_positions[col, 1] + 0.03, label, color="#9c2c13")

    for row, col in assignment:
        ax.plot(
            [row_positions[row, 0], col_positions[col, 0]],
            [row_positions[row, 1], col_positions[col, 1]],
            color="#1d3557",
            linewidth=2.5,
            alpha=0.85,
            zorder=2,
        )

    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    if title is not None:
        ax.set_title(title)
    return ax


def plot_trace_frame(
    frame: TraceFrame,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    cmap: str = "cividis",
) -> plt.Axes:
    values = frame.reduced_cost
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))

    row_labels = _extend_labels(row_labels, values.shape[0], "row_")
    col_labels = _extend_labels(col_labels, values.shape[1], "col_")

    plot_cost_heatmap(
        values,
        assignment=frame.matching,
        row_labels=row_labels,
        col_labels=col_labels,
        ax=ax,
        title=f"Augmentation {frame.augmentation}: {frame.stage}",
        cmap=cmap,
        value_fmt="{:.2f}",
        highlight_zeros=True,
        show_colorbar=False,
        annotation_fontsize=11,
    )

    if frame.used_rows is not None:
        for row, active in enumerate(frame.used_rows):
            if active:
                ax.add_patch(
                    Rectangle(
                        (-0.68, row - 0.5),
                        0.14,
                        1.0,
                        facecolor="#f4d35e",
                        edgecolor="none",
                        alpha=0.8,
                    )
                )

    if frame.used_columns is not None:
        for col, active in enumerate(frame.used_columns):
            if active:
                ax.add_patch(
                    Rectangle(
                        (col - 0.5, -0.68),
                        1.0,
                        0.14,
                        facecolor="#ee964b",
                        edgecolor="none",
                        alpha=0.8,
                    )
                )

    note = frame.note
    if frame.delta is not None:
        note = f"{note}  Delta = {frame.delta:.3f}"
    ax.figure.suptitle(note, fontsize=11, y=0.98)
    ax.figure.tight_layout()
    return ax


def make_trace_animation_html(
    result: HungarianResult,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    interval: int = 1200,
) -> str:
    frames = result.trace
    row_labels = list(row_labels or _default_labels("row_", result.padded_cost.shape[0]))
    col_labels = list(col_labels or _default_labels("col_", result.padded_cost.shape[1]))

    fig, ax = plt.subplots(figsize=(7, 5.5))

    def draw(frame_index: int) -> None:
        ax.clear()
        frame = frames[frame_index]
        plot_cost_heatmap(
            frame.reduced_cost,
            assignment=frame.matching,
            row_labels=row_labels,
            col_labels=col_labels,
            ax=ax,
            title=f"Augmentation {frame.augmentation}: {frame.stage}",
            cmap="viridis",
            value_fmt="{:.2f}",
            highlight_zeros=True,
            show_colorbar=False,
        )

        if frame.used_rows is not None:
            for row, active in enumerate(frame.used_rows):
                if active:
                    ax.add_patch(
                        Rectangle(
                            (-0.68, row - 0.5),
                            0.14,
                            1.0,
                            facecolor="#f4d35e",
                            edgecolor="none",
                            alpha=0.8,
                        )
                    )

        if frame.used_columns is not None:
            for col, active in enumerate(frame.used_columns):
                if active:
                    ax.add_patch(
                        Rectangle(
                            (col - 0.5, -0.68),
                            1.0,
                            0.14,
                            facecolor="#ee964b",
                            edgecolor="none",
                            alpha=0.8,
                        )
                    )

        note = frame.note
        if frame.delta is not None:
            note = f"{note}  Delta = {frame.delta:.3f}"
        fig.suptitle(note, fontsize=11, y=0.98)
        fig.tight_layout()

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=interval, repeat=True)
    fps = max(1, round(1000 / interval))
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=True) as tmp:
        animation.save(tmp.name, writer=PillowWriter(fps=fps))
        tmp.seek(0)
        gif_bytes = tmp.read()
    plt.close(fig)
    gif_base64 = base64.b64encode(gif_bytes).decode("ascii")
    return (
        '<div style="display:flex;justify-content:center;">'
        f'<img src="data:image/gif;base64,{gif_base64}" '
        'style="max-width:100%;height:auto;border:1px solid rgba(128,128,128,0.35);border-radius:8px;" '
        'alt="Hungarian algorithm trace animation" />'
        "</div>"
    )


__all__ = [
    "make_trace_animation_html",
    "plot_bipartite_assignment",
    "plot_cost_heatmap",
    "plot_trace_frame",
    "plot_spatial_assignment",
]
