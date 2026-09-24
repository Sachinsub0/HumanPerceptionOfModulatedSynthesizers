"""Generate publication-ready bar graph of ANOVA variance decompositions.

Reads data/anova_variance_results.tsv, data/anova_variance_results_pooled_source.tsv,
or data/anova_variance_results_pooled_timbre_source.tsv and generates a stacked bar chart
decomposing total variance across factorial components (Distance/Amount, Modulation Type,
Feature, Source, 2-Way Interactions, and Higher-Order Residual), with distinct harmonious
colors for each component.

Dynamically adapts active components and legend depending on whether factors were pooled.
When only 2 factors remain without explicit interaction terms, the residual variance represents
the confounded 2-way interaction and is labeled '2-Way Inter.' with the matching interaction color.

Supports both horizontal and vertical orientations, with configurable white space between
individual loss functions, method groups, human data, bottom bar and x-axis, and between the legend and graph.

Typography & Font Size options:
- --fontsize-labels: font size for entity/method labels (default: 9.0).
- --fontsize-percentages: font size for percentage text inside bar segments (default: 9.0).
- --fontsize-axis-label: optional override for 'Explained Variance (%)' axis label (default: same as labels).
- --fontsize-ticks: optional override for numeric tick values font size (default: same as labels).
- --fontsize-legend: optional override for legend font size (default: 9.0).

Legend & Layout options:
- --legend-rows: number of lines/rows to split the legend across (default: 2).
- --legend-cols: number of columns for the legend (overrides --legend-rows).
- --legend-loc: anchor location for the legend (default: 'lower right').
- --legend-x: horizontal position/offset for legend (default: 1.0, right-aligned up to 100% tick marker).
- --legend-columnspacing: spacing between columns in the legend (default: 1.2).
- --component-spacing: width of vertical white space bar between different components of the same row (default: 0.8).
- --no-group-lines: do not draw dashed lines between groups (human, stft, wavelet, neural).
- --tilt-x / --tilt-x-labels: tilt the x-axis labels at 45 degrees.
- --tilt-y / --tilt-y-labels: tilt the y-axis labels at 45 degrees.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Method ordering matching correlation and ANOVA tables
ENTITIES = [
    ("human", "Human", "human"),
    ("mss_log_lin", "MSS L+L", "group1"),
    ("mss_rev", "MSS Rev.", "group1"),
    ("mfcc", "MFCC", "group1"),
    ("scat1d_log1p", "Scat1D", "group2"),
    ("jtfs_log1p", "JTFS", "group2"),
    ("vggish", "VGGish", "group3"),
    ("encodec48k", "EnCodec", "group3"),
    ("clap2", "MS-CLAP", "group3"),
    ("panns_wavegram_logmel", "PANNs", "group3"),
]

LOSS_FN_ALIASES = {
    "scat1d": "scat1d_log1p",
    "jtfs": "jtfs_log1p",
    "encodec": "encodec48k",
    "encodec48": "encodec48k",
    "encodec24k": "encodec48k",
    "clap": "clap2",
    "panns_wglm": "panns_wavegram_logmel",
}

# The canonical variance components, display labels, and distinct harmonious colors
COMPONENTS = [
    ("distance", "Modulation Amount", "#082a54"),
    ("mod", "Modulation Type", "#d9534f"),
    ("feat", "Timbre Quality", "#3399a1"),
    ("source", "Wavetable Source", "#f39c12"),
    ("two_way", "2-Way Interactions", "#a559aa"),
    ("higher_order", "Higher Order", "#bbbbbb"),
]


def load_variance_data(
    tsv_path: Path,
    distance_label: str = "Distance",
) -> tuple[list[str], dict[str, list[float]], list[str], list[tuple[str, str, str]]]:
    """Load and organize variance components for each entity from TSV.

    Dynamically detects active factors and interactions, omitting pooled factors.
    In an unreplicated 2-factor design, the residual variance represents the 2-way
    interaction and is categorized as '2-Way Inter.'.
    """
    sep = "\t" if tsv_path.suffix in [".tsv", ".txt"] else ","
    df = pd.read_csv(tsv_path, sep=sep)

    if "loss_fn" in df.columns:
        df["canonical_loss"] = df["loss_fn"].map(lambda x: LOSS_FN_ALIASES.get(x, x))

    present_sources = set(df["Source"].dropna().unique())
    has_two_way = any(k.count(":") == 1 for k in present_sources)
    has_three_way = any(k.count(":") == 2 for k in present_sources)
    has_residual = any("Residual" in k for k in present_sources)

    # Determine potential active components dynamically based on input sources
    candidate_components: list[tuple[str, str, str]] = []
    if "rating_stimulus" in present_sources:
        candidate_components.append(("distance", distance_label, "#082a54"))
    if "modulation" in present_sources:
        candidate_components.append(("mod", "Modulation Type", "#d9534f"))
    if "feature" in present_sources:
        candidate_components.append(("feat", "Timbre Quality", "#3399a1"))
    if "source" in present_sources:
        candidate_components.append(("source", "Wavetable Source", "#f39c12"))

    # If explicit 2-way terms exist OR if residual in 2-factor design represents the 2-way interaction
    residual_is_two_way = has_residual and not has_two_way and not has_three_way
    if has_two_way or residual_is_two_way:
        candidate_components.append(("two_way", "2-Way Interactions", "#a559aa"))

    if has_three_way:
        candidate_components.append(("three_way", "3-Way Interactions", "#9b59b6"))

    if has_residual and not residual_is_two_way:
        candidate_components.append(("higher_order", "Higher Order", "#bbbbbb"))

    # Fallback to standard 6 components if no recognized sources
    if not candidate_components:
        candidate_components = list(COMPONENTS)

    labels: list[str] = []
    groups: list[str] = []
    comp_values: dict[str, list[float]] = {k: [] for k, _, _ in candidate_components}

    for canon_key, display_name, grp in ENTITIES:
        match = df[df["canonical_loss"] == canon_key]
        if match.empty:
            match = df[df["loss_fn"] == canon_key]
        if match.empty:
            continue

        labels.append(display_name)
        groups.append(grp)

        t_map = dict(zip(match["Source"], match["pct_var"]))
        two_way_pct = sum(v for k, v in t_map.items() if k.count(":") == 1)
        three_way_pct = sum(v for k, v in t_map.items() if k.count(":") == 2)
        residual_pct = next((v for k, v in t_map.items() if "Residual" in k), 0.0)

        # In an unreplicated 2-factor design, residual is mathematically the 2-way interaction
        if residual_is_two_way:
            two_way_pct = residual_pct

        for key, _, _ in candidate_components:
            if key == "distance":
                comp_values["distance"].append(float(t_map.get("rating_stimulus", 0.0)))
            elif key == "mod":
                comp_values["mod"].append(float(t_map.get("modulation", 0.0)))
            elif key == "feat":
                comp_values["feat"].append(float(t_map.get("feature", 0.0)))
            elif key == "source":
                comp_values["source"].append(float(t_map.get("source", 0.0)))
            elif key == "two_way":
                comp_values["two_way"].append(float(two_way_pct))
            elif key == "three_way":
                comp_values["three_way"].append(float(three_way_pct))
            elif key == "higher_order":
                comp_values["higher_order"].append(float(residual_pct))

    # Keep only components that have non-zero variance in at least one entity
    active_components = [
        c for c in candidate_components
        if any(abs(v) > 1e-4 for v in comp_values[c[0]])
    ]

    return labels, comp_values, groups, active_components


def compute_positions(
    groups: list[str],
    bar_size: float = 0.65,
    bar_spacing: float = 0.35,
    group_spacing: float = 0.35,
    human_spacing: Optional[float] = None,
) -> tuple[np.ndarray, float, list[float]]:
    """Compute center coordinates for each bar and separator positions between groups.

    Args:
        groups: Group identifiers for each entity.
        bar_size: Thickness of each bar.
        bar_spacing: Whitespace gap between consecutive loss function bars.
        group_spacing: Additional whitespace gap between loss function groups.
        human_spacing: Additional whitespace gap between human data and model representations
            (defaults to group_spacing if not specified).

    Returns:
        positions: Coordinate array for bar centers.
        separator_pos: Coordinate midway between Human Listeners and the first model.
        group_separators: List of coordinates midway between all adjacent distinct groups.
    """
    effective_human_spacing = human_spacing if human_spacing is not None else group_spacing
    positions = []
    current_pos = 0.0
    step = bar_size + bar_spacing

    for i, grp in enumerate(groups):
        if i == 0:
            positions.append(0.0)
            current_pos = 0.0
        else:
            delta = step
            if groups[i - 1] == "human" or grp == "human":
                delta += effective_human_spacing
            elif grp != groups[i - 1]:
                delta += group_spacing
            current_pos += delta
            positions.append(current_pos)

    # Compute separator positions between all adjacent distinct groups
    group_separators: list[float] = []
    separator_pos = 0.0
    for i in range(len(groups) - 1):
        if groups[i] != groups[i + 1]:
            sep = (positions[i] + positions[i + 1]) / 2.0
            group_separators.append(sep)
            if groups[i] == "human" or groups[i + 1] == "human":
                separator_pos = sep

    if separator_pos == 0.0 and len(positions) > 1:
        separator_pos = (positions[0] + positions[1]) / 2.0

    return np.array(positions), separator_pos, group_separators


def reorder_legend_left_to_right(
    handles: Sequence,
    labels: Sequence[str],
    ncols: int,
) -> tuple[list, list[str]]:
    """Reorder legend handles and labels so they are read left-to-right, row-by-row.

    By default, matplotlib populates multi-column legends column-by-column (top-to-bottom).
    This rearranges items so that when matplotlib places them into columns, the resulting
    visual layout reads row-by-row from left to right.
    """
    n = len(handles)
    if ncols <= 1 or n <= 1:
        return list(handles), list(labels)

    splits = np.array_split(range(n), ncols)
    col_lens = [len(s) for s in splits]
    nrows = max(col_lens)
    grid = [[None] * ncols for _ in range(nrows)]

    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if r < col_lens[c]:
                grid[r][c] = (handles[idx], labels[idx])
                idx += 1

    reordered_handles = []
    reordered_labels = []
    for c in range(ncols):
        for r in range(col_lens[c]):
            h, l = grid[r][c]
            reordered_handles.append(h)
            reordered_labels.append(l)

    return reordered_handles, reordered_labels


def plot_variance_horizontal(
    labels: list[str],
    comp_values: dict[str, list[float]],
    groups: list[str],
    components: Optional[list[tuple[str, str, str]]] = None,
    output_path: Optional[Path] = None,
    bar_size: float = 0.65,
    bar_spacing: float = 0.35,
    group_spacing: float = 0.35,
    human_spacing: Optional[float] = None,
    bottom_spacing: Optional[float] = None,
    legend_y: Optional[float] = None,
    legend_x: Optional[float] = None,
    legend_loc: Optional[str] = None,
    legend_rows: int = 2,
    legend_ncol: Optional[int] = None,
    legend_columnspacing: Optional[float] = None,
    component_spacing: float = 0.8,
    show_labels: bool = True,
    show_group_lines: bool = True,
    label_threshold: float = 4.5,
    fontsize_labels: float = 9.0,
    fontsize_percentages: float = 9.0,
    fontsize_axis_label: Optional[float] = None,
    fontsize_ticks: Optional[float] = None,
    fontsize_legend: Optional[float] = None,
    tilt_x: bool = False,
    x_rotation: Optional[float] = None,
    tilt_y: bool = False,
    y_rotation: Optional[float] = None,
    title: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """Create a horizontal stacked bar chart of the variance decomposition."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9.0,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    })

    comps_to_plot = components if components is not None else COMPONENTS
    eff_axis_label = fontsize_axis_label if fontsize_axis_label is not None else fontsize_labels
    eff_ticks = fontsize_ticks if fontsize_ticks is not None else fontsize_labels
    eff_legend = fontsize_legend if fontsize_legend is not None else 9.0

    y_pos, separator_pos, group_separators = compute_positions(
        groups=groups,
        bar_size=bar_size,
        bar_spacing=bar_spacing,
        group_spacing=group_spacing,
        human_spacing=human_spacing,
    )

    n_bars = len(labels)
    total_span = y_pos[-1] - y_pos[0]
    fig_height = max(5.5, 6.0 * (total_span / 9.0))
    fig, ax = plt.subplots(figsize=(7.07, fig_height), dpi=dpi)

    cum_left = np.zeros(n_bars)

    for key, name, color in comps_to_plot:
        vals = np.array(comp_values[key])
        rects = ax.barh(
            y_pos,
            vals,
            left=cum_left,
            height=bar_size,
            color=color,
            edgecolor="white" if component_spacing > 0 else "none",
            linewidth=component_spacing,
            label=name,
        )

        if show_labels:
            for i, (val, rect) in enumerate(zip(vals, rects)):
                if val >= label_threshold:
                    x_center = cum_left[i] + val / 2.0
                    y_center = rect.get_y() + rect.get_height() / 2.0
                    val_str = f"{val:.0f}%" if val < 10 else f"{val:.0f}%"
                    ax.text(
                        x_center,
                        y_center,
                        val_str,
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=fontsize_percentages,
                        fontweight="bold",
                    )

        cum_left += vals

    # Format y-axis (invert so Human Listeners is at the top)
    bot_pad = bottom_spacing if bottom_spacing is not None else 0.55
    if bot_pad <= 0.3:
        bot_pad = 0.5 + bot_pad

    ax.set_yticks(y_pos)
    if tilt_y or y_rotation is not None:
        y_rot = y_rotation if y_rotation is not None else 45
        ax.set_yticklabels(labels, rotation=y_rot, ha="right", va="center", fontsize=fontsize_labels, fontweight="bold")
    else:
        ax.set_yticklabels(labels, fontsize=fontsize_labels, fontweight="bold")
    ax.tick_params(axis="y", labelsize=fontsize_labels)
    ax.set_ylim(y_pos[-1] + bar_size * bot_pad, y_pos[0] - bar_size * 0.60)

    # Visual separator dashed lines between groups (Human, STFT, Wavelet, Neural)
    if show_group_lines:
        for sep in group_separators:
            ax.axhline(sep, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)

    # Format x-axis
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Explained Variance (%)", fontsize=eff_axis_label, fontweight="bold", labelpad=8)
    ax.tick_params(axis="x", labelsize=eff_ticks)
    if tilt_x or x_rotation is not None:
        rot = x_rotation if x_rotation is not None else 45
        plt.setp(ax.get_xticklabels(), rotation=rot, ha="right")
    ax.grid(axis="x", linestyle=":", color="#cccccc", alpha=0.7)
    ax.set_axisbelow(True)

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend at the top split across lines (default: right-aligned up to 100% tick marker)
    if legend_ncol is not None:
        ncol = legend_ncol
    elif legend_rows is not None and legend_rows > 0:
        ncol = int(np.ceil(len(comps_to_plot) / legend_rows))
    else:
        ncol = len(comps_to_plot)

    effective_legend_loc = legend_loc if legend_loc is not None else "lower right"
    effective_legend_y = legend_y if legend_y is not None else 1.005
    effective_legend_x = legend_x if legend_x is not None else (1.0 if "right" in effective_legend_loc else 0.0)
    col_spacing = legend_columnspacing if legend_columnspacing is not None else 1.2

    handles, leg_labels = ax.get_legend_handles_labels()
    handles, leg_labels = reorder_legend_left_to_right(handles, leg_labels, ncols=ncol)

    ax.legend(
        handles,
        leg_labels,
        loc=effective_legend_loc,
        bbox_to_anchor=(effective_legend_x, effective_legend_y),
        ncol=ncol,
        frameon=False,
        fontsize=eff_legend,
        columnspacing=col_spacing,
        handlelength=1.2,
        handleheight=0.9,
        borderaxespad=0.0,
    )

    if title:
        fig.suptitle(title, fontsize=9.0, fontweight="bold", y=1.06)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {output_path}")

    if show:
        plt.show()

    return fig


def plot_variance_vertical(
    labels: list[str],
    comp_values: dict[str, list[float]],
    groups: list[str],
    components: Optional[list[tuple[str, str, str]]] = None,
    output_path: Optional[Path] = None,
    bar_size: float = 0.65,
    bar_spacing: float = 0.35,
    group_spacing: float = 0.35,
    human_spacing: Optional[float] = None,
    legend_y: Optional[float] = None,
    legend_x: Optional[float] = None,
    legend_loc: Optional[str] = None,
    legend_rows: int = 2,
    legend_ncol: Optional[int] = None,
    legend_columnspacing: Optional[float] = None,
    component_spacing: float = 0.8,
    show_labels: bool = True,
    show_group_lines: bool = True,
    label_threshold: float = 4.5,
    fontsize_labels: float = 9.0,
    fontsize_percentages: float = 9.0,
    fontsize_axis_label: Optional[float] = None,
    fontsize_ticks: Optional[float] = None,
    fontsize_legend: Optional[float] = None,
    tilt_x: bool = False,
    x_rotation: Optional[float] = None,
    tilt_y: bool = False,
    y_rotation: Optional[float] = None,
    title: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """Create a vertical stacked bar chart of the variance decomposition."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9.0,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    })

    comps_to_plot = components if components is not None else COMPONENTS
    eff_axis_label = fontsize_axis_label if fontsize_axis_label is not None else fontsize_labels
    eff_ticks = fontsize_ticks if fontsize_ticks is not None else fontsize_labels
    eff_legend = fontsize_legend if fontsize_legend is not None else 9.0

    x_pos, separator_pos, group_separators = compute_positions(
        groups=groups,
        bar_size=bar_size,
        bar_spacing=bar_spacing,
        group_spacing=group_spacing,
        human_spacing=human_spacing,
    )

    n_bars = len(labels)
    total_span = x_pos[-1] - x_pos[0]
    fig_width = max(8.5, 11.0 * (total_span / 9.0))
    fig, ax = plt.subplots(figsize=(fig_width, 6.5), dpi=dpi)

    cum_bottom = np.zeros(n_bars)

    for key, name, color in comps_to_plot:
        vals = np.array(comp_values[key])
        rects = ax.bar(
            x_pos,
            vals,
            bottom=cum_bottom,
            width=bar_size,
            color=color,
            edgecolor="white" if component_spacing > 0 else "none",
            linewidth=component_spacing,
            label=name,
        )

        if show_labels:
            for i, (val, rect) in enumerate(zip(vals, rects)):
                if val >= label_threshold:
                    x_center = rect.get_x() + rect.get_width() / 2.0
                    y_center = cum_bottom[i] + val / 2.0
                    val_str = f"{val:.0f}%" if val < 10 else f"{val:.0f}%"
                    ax.text(
                        x_center,
                        y_center,
                        val_str,
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=fontsize_percentages,
                        fontweight="bold",
                    )

        cum_bottom += vals

    # Format x-axis
    rot = x_rotation if x_rotation is not None else (45 if tilt_x else 35)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=rot, ha="right", fontsize=fontsize_labels, fontweight="bold")
    ax.tick_params(axis="x", labelsize=fontsize_labels)
    ax.set_xlim(x_pos[0] - bar_size * 0.8, x_pos[-1] + bar_size * 0.8)

    # Visual separator dashed lines between groups (Human, STFT, Wavelet, Neural)
    if show_group_lines:
        for sep in group_separators:
            ax.axvline(sep, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)

    # Format y-axis
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("Explained Variance (%)", fontsize=eff_axis_label, fontweight="bold", labelpad=8)
    ax.tick_params(axis="y", labelsize=eff_ticks)
    if tilt_y or y_rotation is not None:
        y_rot = y_rotation if y_rotation is not None else 45
        plt.setp(ax.get_yticklabels(), rotation=y_rot, ha="right", va="center")
    ax.grid(axis="y", linestyle=":", color="#cccccc", alpha=0.7)
    ax.set_axisbelow(True)

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend at the top split across lines (default: right-aligned up to 100% tick marker)
    if legend_ncol is not None:
        ncol = legend_ncol
    elif legend_rows is not None and legend_rows > 0:
        ncol = int(np.ceil(len(comps_to_plot) / legend_rows))
    else:
        ncol = len(comps_to_plot)

    effective_legend_loc = legend_loc if legend_loc is not None else "lower right"
    effective_legend_y = legend_y if legend_y is not None else 1.01
    effective_legend_x = legend_x if legend_x is not None else (1.0 if "right" in effective_legend_loc else 0.0)
    col_spacing = legend_columnspacing if legend_columnspacing is not None else 1.2

    handles, leg_labels = ax.get_legend_handles_labels()
    handles, leg_labels = reorder_legend_left_to_right(handles, leg_labels, ncols=ncol)

    ax.legend(
        handles,
        leg_labels,
        loc=effective_legend_loc,
        bbox_to_anchor=(effective_legend_x, effective_legend_y),
        ncol=ncol,
        frameon=False,
        fontsize=eff_legend,
        columnspacing=col_spacing,
        handlelength=1.2,
        handleheight=0.9,
        borderaxespad=0.0,
    )

    if title:
        fig.suptitle(title, fontsize=9.0, fontweight="bold", y=1.06)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {output_path}")

    if show:
        plt.show()

    return fig


def resolve_file_path(path_str: str) -> Path:
    """Resolve file path relative to current working dir, repo root, or script location."""
    p = Path(path_str).expanduser()
    if p.exists():
        return p.resolve()
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate = (repo_root / path_str).resolve()
    if candidate.exists():
        return candidate
    candidate_script = (Path(__file__).resolve().parent / path_str).resolve()
    if candidate_script.exists():
        return candidate_script

    # Fallback mappings for pooled variance files
    if path_str.endswith("anova_variance_results_pooled.tsv"):
        alt = candidate.with_name("anova_variance_results_pooled_timbre_source.tsv")
        if alt.exists():
            return alt

    return p.resolve()


def resolve_output_path(path_str: str) -> Path:
    """Resolve output file path properly relative to cwd, repo root, or script location."""
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parent.parent.parent
    # If path_str was written relative to script dir (../../out/...) but run from repo root
    if str(path_str).startswith("../../"):
        rel_stripped = str(path_str)[6:]
        candidate_repo = (repo_root / rel_stripped).resolve()
        if candidate_repo.parent.exists():
            return candidate_repo
    candidate_cwd = p.resolve()
    if candidate_cwd.parent.exists():
        return candidate_cwd
    candidate_script = (Path(__file__).resolve().parent / path_str).resolve()
    if candidate_script.parent.exists():
        return candidate_script
    return (repo_root / path_str).resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Generate stacked bar chart of ANOVA variance decompositions."
    )
    parser.add_argument(
        "input",
        nargs="?",
        # default="data/anova_variance_results.tsv",
        default="data/anova_variance_results_pooled_timbre_source.tsv",
        # default="data/anova_variance_results_pooled_source.tsv",
        help="Path to anova_variance_results.tsv, anova_variance_results_pooled_timbre_source.tsv, or anova_variance_results_pooled_source.tsv (default: data/anova_variance_results.tsv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to save figure image. If omitted, defaults to out/figure_variance.pdf or derives name from input TSV. Supports .png, .pdf, .svg, etc.",
    )
    parser.add_argument(
        "--distance-label",
        default="Modulation Amount",
        help="Display label for the rating_stimulus / distance variance component (default: Distance; e.g. 'Amount').",
    )
    parser.add_argument(
        "--amount",
        action="store_const",
        dest="distance_label",
        const="Amount",
        help="Shortcut to set distance variance component label to 'Amount'.",
    )
    parser.add_argument(
        "--orientation",
        "--dir",
        choices=["horizontal", "vertical", "h", "v"],
        default="horizontal",
        help="Orientation of the bars: 'horizontal' (default) or 'vertical'.",
    )
    parser.add_argument(
        "--horizontal",
        action="store_true",
        help="Shortcut to set horizontal bar orientation.",
    )
    parser.add_argument(
        "--vertical",
        action="store_true",
        help="Shortcut to set vertical bar orientation.",
    )
    parser.add_argument(
        "--spacing",
        "--bar-spacing",
        type=float,
        default=0.0,
        dest="bar_spacing",
        help="Whitespace gap between adjacent loss function bars (default: 0.0).",
    )
    parser.add_argument(
        "--group-spacing",
        type=float,
        default=0.2,
        dest="group_spacing",
        help="Additional whitespace gap between loss function groups (default: 0.2).",
    )
    parser.add_argument(
        "--human-spacing",
        "--spacing-human",
        "--human-gap",
        "--gap-human",
        type=float,
        default=0.2,
        dest="human_spacing",
        help="Additional whitespace gap between human data and model representations (default: same as --group-spacing, i.e. 0.2).",
    )
    parser.add_argument(
        "--bottom-spacing",
        "--bottom-pad",
        "--bottom-margin",
        type=float,
        default=0.05,
        dest="bottom_spacing",
        help="Whitespace padding between the lowest bar and the x-axis in horizontal mode (default: 0.55, where 0.5 is the bar boundary).",
    )
    parser.add_argument(
        "--legend-spacing",
        "--legend-gap",
        "--legend-y",
        type=float,
        default=1.0,
        dest="legend_y",
        help="Vertical position/offset of legend above the graph (default: 1.005 for horizontal, 1.01 for vertical).",
    )
    parser.add_argument(
        "--legend-loc",
        default="lower right",
        dest="legend_loc",
        help="Legend placement anchor location (default: 'lower right').",
    )
    parser.add_argument(
        "--legend-x",
        type=float,
        default=None,
        dest="legend_x",
        help="Horizontal position/offset of legend (default: 1.0 to right-align up to 100%% tick marker).",
    )
    parser.add_argument(
        "--legend-columnspacing",
        "--legend-col-spacing",
        type=float,
        default=None,
        dest="legend_columnspacing",
        help="Spacing between columns in the legend (default: 1.2).",
    )
    parser.add_argument(
        "--component-spacing",
        "--comp-spacing",
        "--segment-spacing",
        "--component-border",
        "--component-width",
        type=float,
        default=2,
        dest="component_spacing",
        help="Width of the white separator line between adjacent components of the same bar (default: 0.8).",
    )
    parser.add_argument(
        "--no-group-lines",
        action="store_true",
        help="Do not draw dashed separator lines between groups (human, stft, wavelet, neural).",
    )
    parser.add_argument(
        "--bar-size",
        "--bar-thickness",
        "--bar-width",
        type=float,
        default=0.7,
        dest="bar_size",
        help="Thickness of individual bars (default: 0.65).",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not display numerical percentage labels on bar segments.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.5,
        help="Minimum percentage required to render label inside a bar segment (default: 4.5).",
    )
    font_size = 16
    parser.add_argument(
        "--fontsize-labels",
        "--font-size-labels",
        "--labels-fontsize",
        "--fontsize-label",
        "--font-size-label",
        type=float,
        default=font_size,
        dest="fontsize_labels",
        help="Font size for entity/method labels (default: 9.0).",
    )
    parser.add_argument(
        "--fontsize-percentages",
        "--font-size-percentages",
        "--percentages-fontsize",
        "--fontsize-pct",
        "--font-size-pct",
        "--pct-fontsize",
        type=float,
        default=font_size - 4,
        dest="fontsize_percentages",
        help="Font size for percentage text inside bar segments (default: 9.0).",
    )
    parser.add_argument(
        "--fontsize-axis-label",
        "--font-size-axis-label",
        "--axis-label-fontsize",
        type=float,
        default=None,
        dest="fontsize_axis_label",
        help="Optional override for 'Explained Variance (%%)' axis label font size (default: same as --fontsize-labels).",
    )
    parser.add_argument(
        "--fontsize-ticks",
        "--font-size-ticks",
        "--tick-fontsize",
        "--ticks-fontsize",
        type=float,
        default=font_size - 4,
        dest="fontsize_ticks",
        help="Optional override for numeric tick values font size (default: same as --fontsize-labels).",
    )
    parser.add_argument(
        "--fontsize-legend",
        "--font-size-legend",
        "--legend-fontsize",
        type=float,
        default=font_size - 3,
        dest="fontsize_legend",
        help="Optional override for legend font size (default: 9.0).",
    )
    parser.add_argument(
        "--legend-rows",
        type=int,
        # default=2,
        default=1,
        dest="legend_rows",
        help="Number of lines/rows to split the legend across (default: 2).",
    )
    parser.add_argument(
        "--legend-cols",
        "--legend-ncol",
        type=int,
        default=None,
        dest="legend_cols",
        help="Number of columns for the legend (overrides --legend-rows).",
    )
    parser.add_argument(
        "--tilt-x",
        "--tilt-x-labels",
        "--tilt-x-45",
        "--tilt-45",
        "--rotate-x",
        action="store_true",
        dest="tilt_x",
        help="Tilt x-axis labels at 45 degrees.",
    )
    parser.add_argument(
        "--x-rotation",
        "--x-rot",
        type=float,
        default=None,
        dest="x_rotation",
        help="Custom rotation angle for x-axis labels in degrees (default: 45 if --tilt-x, else 35 for vertical / 0 for horizontal).",
    )
    parser.add_argument(
        "--tilt-y",
        "--tilt-y-labels",
        "--tilt-y-45",
        "--tilt-45-y",
        "--rotate-y",
        action="store_true",
        # default=True,
        dest="tilt_y",
        help="Tilt y-axis labels at 45 degrees.",
    )
    parser.add_argument(
        "--y-rotation",
        "--y-rot",
        type=float,
        default=None,
        dest="y_rotation",
        help="Custom rotation angle for y-axis labels in degrees (default: 45 if --tilt-y, else 0).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in dots per inch for raster export (default: 300).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title to display above the plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plot interactively in a window (only save to file).",
    )
    args = parser.parse_args()

    input_path = resolve_file_path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Error: ANOVA variance results file not found at: {input_path}\n")
        sys.exit(1)

    labels, comp_values, groups, active_components = load_variance_data(
        input_path,
        distance_label=args.distance_label,
    )

    # Determine orientation
    orientation = args.orientation.lower()
    if args.vertical:
        orientation = "vertical"
    elif args.horizontal:
        orientation = "horizontal"

    # Resolve legend_y
    legend_y = args.legend_y
    if legend_y is not None and legend_y <= 0.5:
        legend_y = 1.0 + legend_y

    # Resolve output path
    repo_root = Path(__file__).resolve().parent.parent.parent
    if args.output:
        out_path = resolve_output_path(args.output)
    else:
        stem = input_path.stem
        if stem == "anova_variance_results":
            out_path = (repo_root / "out" / "figure_variance.pdf").resolve()
        elif stem.startswith("anova_variance_results_"):
            suffix = stem[len("anova_variance_results_"):]
            out_path = (repo_root / "out" / f"figure_variance_{suffix}.pdf").resolve()
        else:
            out_path = (repo_root / "out" / f"figure_{stem}.pdf").resolve()

    should_show = not args.no_show

    if orientation in ["horizontal", "h"]:
        plot_variance_horizontal(
            labels=labels,
            comp_values=comp_values,
            groups=groups,
            components=active_components,
            output_path=out_path,
            bar_size=args.bar_size,
            bar_spacing=args.bar_spacing,
            group_spacing=args.group_spacing,
            human_spacing=args.human_spacing,
            bottom_spacing=args.bottom_spacing,
            legend_y=legend_y,
            legend_x=args.legend_x,
            legend_loc=args.legend_loc,
            legend_rows=args.legend_rows,
            legend_ncol=args.legend_cols,
            legend_columnspacing=args.legend_columnspacing,
            component_spacing=args.component_spacing,
            show_labels=not args.no_labels,
            show_group_lines=not args.no_group_lines,
            label_threshold=args.threshold,
            fontsize_labels=args.fontsize_labels,
            fontsize_percentages=args.fontsize_percentages,
            fontsize_axis_label=args.fontsize_axis_label,
            fontsize_ticks=args.fontsize_ticks,
            fontsize_legend=args.fontsize_legend,
            tilt_x=args.tilt_x,
            x_rotation=args.x_rotation,
            tilt_y=args.tilt_y,
            y_rotation=args.y_rotation,
            title=args.title,
            dpi=args.dpi,
            show=should_show,
        )
    else:
        plot_variance_vertical(
            labels=labels,
            comp_values=comp_values,
            groups=groups,
            components=active_components,
            output_path=out_path,
            bar_size=args.bar_size,
            bar_spacing=args.bar_spacing,
            group_spacing=args.group_spacing,
            human_spacing=args.human_spacing,
            legend_y=legend_y,
            legend_x=args.legend_x,
            legend_loc=args.legend_loc,
            legend_rows=args.legend_rows,
            legend_ncol=args.legend_cols,
            legend_columnspacing=args.legend_columnspacing,
            component_spacing=args.component_spacing,
            show_labels=not args.no_labels,
            show_group_lines=not args.no_group_lines,
            label_threshold=args.threshold,
            fontsize_labels=args.fontsize_labels,
            fontsize_percentages=args.fontsize_percentages,
            fontsize_axis_label=args.fontsize_axis_label,
            fontsize_ticks=args.fontsize_ticks,
            fontsize_legend=args.fontsize_legend,
            tilt_x=args.tilt_x,
            x_rotation=args.x_rotation,
            tilt_y=args.tilt_y,
            y_rotation=args.y_rotation,
            title=args.title,
            dpi=args.dpi,
            show=should_show,
        )


if __name__ == "__main__":
    main()
