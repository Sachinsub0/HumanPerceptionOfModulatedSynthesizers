"""Generate publication-ready bar graph of ANOVA variance decompositions.

Reads data/anova_variance_results.tsv and generates a stacked bar chart
decomposing total variance across 6 components (Distance, Modulation Type,
Feature, Source, 2-Way Interactions, and Higher-Order Residual), with distinct
colors for each component. Supports both horizontal and vertical orientations,
with configurable white space between individual loss functions and method groups.
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
    ("human", "Human Listeners", "human"),
    ("mss_log_lin", "MSS Log + Lin.", "group1"),
    ("mss_rev", "MSS Revisited", "group1"),
    ("mfcc", "MFCC", "group1"),
    ("scat1d_log1p", "Scat1D", "group2"),
    ("jtfs_log1p", "JTFS", "group2"),
    ("vggish", "VGGish", "group3"),
    ("encodec48k", "EnCodec 48kHz", "group3"),
    ("clap2", "MS-CLAP", "group3"),
    ("panns_wavegram_logmel", "PANNs WGLM", "group3"),
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

# The 6 variance components, display labels, and distinct harmonious colors
COMPONENTS = [
    ("distance", "Distance", "#2b5c8f"),        # Deep Steel Blue
    ("mod", "Mod. Type", "#3399a1"),         # Cyan / Teal
    ("feat", "Feature", "#f39c12"),          # Amber / Orange
    ("source", "Source", "#d9534f"),         # Coral / Crimson
    ("two_way", "2-Way Inter.", "#8e44ad"),  # Amethyst Purple
    ("higher_order", "Higher Order", "#7f8c8d"),  # Slate Gray
]


def load_variance_data(
    tsv_path: Path,
) -> tuple[list[str], dict[str, list[float]], list[str]]:
    """Load and organize variance components for each entity from TSV."""
    sep = "\t" if tsv_path.suffix in [".tsv", ".txt"] else ","
    df = pd.read_csv(tsv_path, sep=sep)

    if "loss_fn" in df.columns:
        df["canonical_loss"] = df["loss_fn"].map(lambda x: LOSS_FN_ALIASES.get(x, x))

    labels: list[str] = []
    groups: list[str] = []
    comp_values: dict[str, list[float]] = {k: [] for k, _, _ in COMPONENTS}

    for canon_key, display_name, grp in ENTITIES:
        match = df[df["canonical_loss"] == canon_key]
        if match.empty:
            match = df[df["loss_fn"] == canon_key]
        if match.empty:
            continue

        labels.append(display_name)
        groups.append(grp)

        t_map = dict(zip(match["Source"], match["pct_var"]))
        two_way_pct = sum(v for k, v in t_map.items() if ":" in k)
        higher_order_pct = t_map.get("Residual (pooled)", 0.0)

        comp_values["distance"].append(float(t_map.get("rating_stimulus", 0.0)))
        comp_values["mod"].append(float(t_map.get("modulation", 0.0)))
        comp_values["feat"].append(float(t_map.get("feature", 0.0)))
        comp_values["source"].append(float(t_map.get("source", 0.0)))
        comp_values["two_way"].append(float(two_way_pct))
        comp_values["higher_order"].append(float(higher_order_pct))

    return labels, comp_values, groups


def compute_positions(
    groups: list[str],
    bar_size: float = 0.65,
    bar_spacing: float = 0.35,
    group_spacing: float = 0.35,
) -> tuple[np.ndarray, float]:
    """Compute center coordinates for each bar and the human separator line position.

    Args:
        groups: Group identifiers for each entity.
        bar_size: Thickness of each bar.
        bar_spacing: Whitespace gap between consecutive loss function bars.
        group_spacing: Additional whitespace gap between loss function groups.

    Returns:
        positions: Coordinate array for bar centers.
        separator_pos: Coordinate midway between Human Listeners and the first model.
    """
    positions = []
    current_pos = 0.0
    step = bar_size + bar_spacing
    separator_pos = step / 2.0

    for i, grp in enumerate(groups):
        if i == 0:
            positions.append(0.0)
            current_pos = 0.0
        elif i == 1:
            current_pos += step
            positions.append(current_pos)
            separator_pos = current_pos / 2.0
        else:
            delta = step
            if grp != groups[i - 1]:
                delta += group_spacing
            current_pos += delta
            positions.append(current_pos)

    return np.array(positions), separator_pos


def plot_variance_horizontal(
    labels: list[str],
    comp_values: dict[str, list[float]],
    groups: list[str],
    output_path: Optional[Path] = None,
    bar_size: float = 0.65,
    bar_spacing: float = 0.35,
    group_spacing: float = 0.35,
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

    y_pos, separator_pos = compute_positions(
        groups=groups,
        bar_size=bar_size,
        bar_spacing=bar_spacing,
        group_spacing=group_spacing,
    )

    n_bars = len(labels)
    total_span = y_pos[-1] - y_pos[0]
    fig_height = max(5.5, 6.0 * (total_span / 9.0))
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=dpi)

    cum_left = np.zeros(n_bars)

    for key, name, color in COMPONENTS:
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
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_ylim(y_pos[-1] + bar_size * 0.8, y_pos[0] - bar_size * 0.8)

    # Visual separator line between Human Listeners and model representations
    if len(labels) > 1:
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

    # Legend at the top
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=6,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.2,
        handlelength=1.2,
        handleheight=0.9,
    )

    if title:
        fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.08)

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
    output_path: Optional[Path] = None,
    bar_size: float = 0.65,
    bar_spacing: float = 0.35,
    group_spacing: float = 0.35,
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

    x_pos, separator_pos = compute_positions(
        groups=groups,
        bar_size=bar_size,
        bar_spacing=bar_spacing,
        group_spacing=group_spacing,
    )

    n_bars = len(labels)
    total_span = x_pos[-1] - x_pos[0]
    fig_width = max(8.5, 11.0 * (total_span / 9.0))
    fig, ax = plt.subplots(figsize=(fig_width, 6.5), dpi=dpi)

    cum_bottom = np.zeros(n_bars)

    for key, name, color in COMPONENTS:
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
    if len(labels) > 1:
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

    # Legend at the top
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=6,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.2,
        handlelength=1.2,
        handleheight=0.9,
    )

    if title:
        fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.08)

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
    return p.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Generate stacked bar chart of ANOVA variance decompositions."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/anova_variance_results.tsv",
        help="Path to anova_variance_results.tsv (default: data/anova_variance_results.tsv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="../../out/figure_variance.png",
        help="Path to save figure image (default: figures/figure_variance.png). Supports .png, .pdf, .svg, etc.",
    )
    parser.add_argument(
        "--orientation",
        "--dir",
        choices=["horizontal", "vertical", "h", "v"],
        default="horizontal",
        # default="vertical",
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
        default=0.10,
        dest="bar_spacing",
        help="Whitespace gap between adjacent loss function bars (default: 0.35).",
    )
    parser.add_argument(
        "--group-spacing",
        type=float,
        default=0.0,
        dest="group_spacing",
        help="Additional whitespace gap between loss function groups (default: 0.35).",
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

    labels, comp_values, groups = load_variance_data(input_path)

    # Determine orientation
    orientation = args.orientation.lower()
    if args.vertical:
        orientation = "vertical"
    elif args.horizontal:
        orientation = "horizontal"

    out_path = Path(args.output).expanduser().resolve() if args.output else None
    should_show = not args.no_show

    if orientation in ["horizontal", "h"]:
        plot_variance_horizontal(
            labels=labels,
            comp_values=comp_values,
            groups=groups,
            output_path=out_path,
            bar_size=args.bar_size,
            bar_spacing=args.bar_spacing,
            group_spacing=args.group_spacing,
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
            output_path=out_path,
            bar_size=args.bar_size,
            bar_spacing=args.bar_spacing,
            group_spacing=args.group_spacing,
            show_labels=not args.no_labels,
            label_threshold=args.threshold,
            title=args.title,
            dpi=args.dpi,
            show=should_show,
        )


if __name__ == "__main__":
    main()
