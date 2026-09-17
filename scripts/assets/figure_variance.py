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
    ("human", "Human\nListeners", "human"),
    ("mss_log_lin", "MSS Log\n+ Linear", "group1"),
    ("mss_rev", "MSS Rev.", "group1"),
    ("mfcc", "MFCC", "group1"),
    ("scat1d_log1p", "Scat1D", "group2"),
    ("jtfs_log1p", "JTFS", "group2"),
    ("vggish", "VGGish", "group3"),
    ("encodec48k", "EnCodec\n48 kHz", "group3"),
    ("clap2", "MS-CLAP", "group3"),
    ("panns_wavegram_logmel", "PANNs\nWGLM", "group3"),
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
    ("distance", "Distance", "#2b5c8f"),        # Deep Steel Blue
    ("mod", "Modulation Type", "#3399a1"),         # Cyan / Teal
    ("feat", "Timbre Quality", "#f39c12"),          # Amber / Orange
    ("source", "Source", "#d9534f"),         # Coral / Crimson
    ("two_way", "2-Way Interactions", "#8e44ad"),  # Amethyst Purple
    ("higher_order", "Higher Order", "#7f8c8d"),  # Slate Gray
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
        candidate_components.append(("distance", distance_label, "#2b5c8f"))
    if "modulation" in present_sources:
        candidate_components.append(("mod", "Modulation Type", "#3399a1"))
    if "feature" in present_sources:
        candidate_components.append(("feat", "Timbre Quality", "#f39c12"))
    if "source" in present_sources:
        candidate_components.append(("source", "Source", "#d9534f"))

    # If explicit 2-way terms exist OR if residual in 2-factor design represents the 2-way interaction
    residual_is_two_way = has_residual and not has_two_way and not has_three_way
    if has_two_way or residual_is_two_way:
        candidate_components.append(("two_way", "2-Way Interactions", "#8e44ad"))

    if has_three_way:
        candidate_components.append(("three_way", "3-Way Interactions", "#9b59b6"))

    if has_residual and not residual_is_two_way:
        candidate_components.append(("higher_order", "Higher Order", "#7f8c8d"))

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
) -> tuple[np.ndarray, float]:
    """Compute center coordinates for each bar and the human separator line position.

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
    """
    effective_human_spacing = human_spacing if human_spacing is not None else group_spacing
    positions = []
    current_pos = 0.0
    step = bar_size + bar_spacing
    separator_pos = (step + effective_human_spacing) / 2.0

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

    if len(positions) > 1 and "human" in groups:
        human_idx = groups.index("human")
        if human_idx == 0:
            separator_pos = (positions[0] + positions[1]) / 2.0
        elif human_idx == len(groups) - 1:
            separator_pos = (positions[-2] + positions[-1]) / 2.0
        else:
            separator_pos = (positions[human_idx] + positions[human_idx + 1]) / 2.0
    elif len(positions) > 1:
        separator_pos = (positions[0] + positions[1]) / 2.0

    return np.array(positions), separator_pos


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
    show_labels: bool = True,
    label_threshold: float = 4.5,
    title: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """Create a horizontal stacked bar chart of the variance decomposition."""
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    })

    comps_to_plot = components if components is not None else COMPONENTS

    y_pos, separator_pos = compute_positions(
        groups=groups,
        bar_size=bar_size,
        bar_spacing=bar_spacing,
        group_spacing=group_spacing,
        human_spacing=human_spacing,
    )

    n_bars = len(labels)
    total_span = y_pos[-1] - y_pos[0]
    fig_height = max(5.5, 6.0 * (total_span / 9.0))
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=dpi)

    cum_left = np.zeros(n_bars)

    for key, name, color in comps_to_plot:
        vals = np.array(comp_values[key])
        rects = ax.barh(
            y_pos,
            vals,
            left=cum_left,
            height=bar_size,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=name,
        )

        if show_labels:
            for i, (val, rect) in enumerate(zip(vals, rects)):
                if val >= label_threshold:
                    x_center = cum_left[i] + val / 2.0
                    y_center = rect.get_y() + rect.get_height() / 2.0
                    val_str = f"{val:.1f}%" if val < 10 else f"{val:.0f}%"
                    ax.text(
                        x_center,
                        y_center,
                        val_str,
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8.5,
                        fontweight="bold",
                    )

        cum_left += vals

    # Format y-axis (invert so Human Listeners is at the top)
    bot_pad = bottom_spacing if bottom_spacing is not None else 0.55
    if bot_pad <= 0.3:
        bot_pad = 0.5 + bot_pad

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_ylim(y_pos[-1] + bar_size * bot_pad, y_pos[0] - bar_size * 0.60)

    # Visual separator line between Human Listeners and model representations
    if len(labels) > 1 and "human" in groups:
        ax.axhline(separator_pos, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)

    # Format x-axis
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Explained Variance (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.grid(axis="x", linestyle=":", color="#cccccc", alpha=0.7)
    ax.set_axisbelow(True)

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend at the top dynamically scaled to number of components
    ncol = len(comps_to_plot)
    effective_legend_y = legend_y if legend_y is not None else 1.005
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, effective_legend_y),
        ncol=ncol,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.2,
        handlelength=1.2,
        handleheight=0.9,
        borderaxespad=0.2,
    )

    if title:
        fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.06)

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
    show_labels: bool = True,
    label_threshold: float = 4.5,
    title: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """Create a vertical stacked bar chart of the variance decomposition."""
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    })

    comps_to_plot = components if components is not None else COMPONENTS

    x_pos, separator_pos = compute_positions(
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
            edgecolor="white",
            linewidth=0.8,
            label=name,
        )

        if show_labels:
            for i, (val, rect) in enumerate(zip(vals, rects)):
                if val >= label_threshold:
                    x_center = rect.get_x() + rect.get_width() / 2.0
                    y_center = cum_bottom[i] + val / 2.0
                    val_str = f"{val:.1f}%" if val < 10 else f"{val:.0f}%"
                    ax.text(
                        x_center,
                        y_center,
                        val_str,
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8.0,
                        fontweight="bold",
                    )

        cum_bottom += vals

    # Format x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_xlim(x_pos[0] - bar_size * 0.8, x_pos[-1] + bar_size * 0.8)

    # Visual separator line between Human Listeners and model representations
    if len(labels) > 1 and "human" in groups:
        ax.axvline(separator_pos, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)

    # Format y-axis
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("Explained Variance (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.grid(axis="y", linestyle=":", color="#cccccc", alpha=0.7)
    ax.set_axisbelow(True)

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend at the top dynamically scaled to number of components
    ncol = len(comps_to_plot)
    effective_legend_y = legend_y if legend_y is not None else 1.01
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, effective_legend_y),
        ncol=ncol,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.2,
        handlelength=1.2,
        handleheight=0.9,
        borderaxespad=0.25,
    )

    if title:
        fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.06)

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
        help="Path to save figure image. If omitted, defaults to out/figure_variance.png or derives name from input TSV. Supports .png, .pdf, .svg, etc.",
    )
    parser.add_argument(
        "--distance-label",
        default="Distance",
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
        default=0.1,
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
        default=0.2,
        dest="bottom_spacing",
        help="Whitespace padding between the lowest bar and the x-axis in horizontal mode (default: 0.55, where 0.5 is the bar boundary).",
    )
    parser.add_argument(
        "--legend-spacing",
        "--legend-gap",
        "--legend-y",
        type=float,
        default=None,
        dest="legend_y",
        help="Vertical position/offset of legend above the graph (default: 1.005 for horizontal, 1.01 for vertical).",
    )
    parser.add_argument(
        "--bar-size",
        "--bar-thickness",
        "--bar-width",
        type=float,
        default=0.65,
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
        default=4.5,
        help="Minimum percentage required to render label inside a bar segment (default: 4.5).",
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
            show_labels=not args.no_labels,
            label_threshold=args.threshold,
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
            show_labels=not args.no_labels,
            label_threshold=args.threshold,
            title=args.title,
            dpi=args.dpi,
            show=should_show,
        )


if __name__ == "__main__":
    main()
