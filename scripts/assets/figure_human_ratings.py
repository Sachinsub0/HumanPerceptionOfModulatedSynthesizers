"""Generate publication-ready 3-panel figure of human listening test ratings.

Reads postprocessed human study data (data/listening_test_responses_postprocessed.tsv)
and creates three 1:1 aspect ratio subplots side by side for Amplitude, Frequency (Hz),
and Irregularity (%). Plots mean perceived difference ratings and standard deviation
error bars, with reference point as a square and remaining four points as circles.
Includes a linear line of best fit in blue and displays the R^2 value in bold blue in
the upper-left corner of each panel. Zero excess outer whitespace, clean padding for
axis labels, and clear gridlines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Modulation conditions and publication-ready subplot configurations
MODULATION_PANELS = [
    {
        "key": "amp",
        "title": "Amplitude",
        "tick_labels": ["0.1", "0.3", "0.5", "0.7", "0.9"],
    },
    {
        "key": "freq",
        "title": "Frequency (Hz)",
        "tick_labels": ["0.25", "0.5", "1", "2", "4"],
    },
    {
        "key": "reg",
        "title": "Irregularity (%)",
        "tick_labels": ["0", "12.5", "25", "37.5", "50"],
    },
]

STIMULUS_ORDER = [
    "reference",
    "condition_a",
    "condition_b",
    "condition_c",
    "condition_d",
]


def load_human_ratings(data_path: Path) -> pd.DataFrame:
    """Load human listening test data and extract modulation category."""
    sep = "\t" if data_path.suffix in [".tsv", ".txt"] else ","
    df = pd.read_csv(data_path, sep=sep)

    if "modulation" not in df.columns:
        if "trial_id" in df.columns:
            split_cols = df["trial_id"].str.split("_", expand=True)
            df["modulation"] = split_cols[0]
            df["feature"] = split_cols[1]
            df["source"] = split_cols[2]
        else:
            raise ValueError("Input data must contain 'trial_id' or 'modulation' column.")

    return df


def plot_human_ratings(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    plot_size: float = 2.85,
    gap: float = 0.10,
    line_color: str = "#2a78d6",
    marker_color: str = "#111111",
    r2_decimals: int = 2,
    fit_target: str = "means",
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """Create 3-panel figure of human ratings across modulation conditions."""
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
    })

    # Exact physical geometry in inches ensuring labels have full clearance without clipping
    margin_left = 0.62    # Y-axis label and tick numbers
    margin_right = 0.04   # Right canvas boundary
    margin_bottom = 0.46  # X-axis ticks and "Modulation amount" label
    margin_top = 0.22     # Panel titles

    fig_w = margin_left + 3 * plot_size + 2 * gap + margin_right
    fig_h = margin_bottom + plot_size + margin_top

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    axes: list[plt.Axes] = []
    for i in range(3):
        left_in = margin_left + i * (plot_size + gap)
        ax = fig.add_axes([
            left_in / fig_w,
            margin_bottom / fig_h,
            plot_size / fig_w,
            plot_size / fig_h,
        ])
        axes.append(ax)

    x_indices = np.arange(len(STIMULUS_ORDER))

    for ax, panel in zip(axes, MODULATION_PANELS):
        mod_key = panel["key"]
        sub_df = df[df["modulation"] == mod_key]

        # Calculate mean and std for each stimulus amount
        means = np.array([
            sub_df[sub_df["rating_stimulus"] == s]["rating_score"].mean()
            for s in STIMULUS_ORDER
        ])
        stds = np.array([
            sub_df[sub_df["rating_stimulus"] == s]["rating_score"].std()
            for s in STIMULUS_ORDER
        ])

        # Linear regression for line of best fit
        if fit_target == "all":
            stim_to_x = {s: i for i, s in enumerate(STIMULUS_ORDER)}
            x_all = sub_df["rating_stimulus"].map(stim_to_x).values
            y_all = sub_df["rating_score"].values
            valid_mask = ~np.isnan(x_all) & ~np.isnan(y_all)
            reg = linregress(x_all[valid_mask], y_all[valid_mask])
        else:
            reg = linregress(x_indices, means)

        r2 = reg.rvalue ** 2

        # Plot thicker line of best fit spanning the domain
        x_line = np.linspace(-0.25, 4.25, 100)
        y_line = reg.slope * x_line + reg.intercept
        ax.plot(
            x_line,
            y_line,
            color=line_color,
            linewidth=2.4,
            linestyle="-",
            zorder=2,
            label="Linear fit",
        )

        # Reference stimulus: square marker with thicker error bars
        ax.errorbar(
            x_indices[0],
            means[0],
            yerr=stds[0],
            fmt="s",
            color=marker_color,
            ecolor=marker_color,
            elinewidth=1.8,
            capsize=4.5,
            capthick=1.8,
            markersize=7.0,
            markerfacecolor=marker_color,
            markeredgecolor=marker_color,
            markeredgewidth=1.2,
            zorder=4,
        )

        # Non-reference stimulus points (conditions a-d): circle markers with thicker error bars
        ax.errorbar(
            x_indices[1:],
            means[1:],
            yerr=stds[1:],
            fmt="o",
            color=marker_color,
            ecolor=marker_color,
            elinewidth=1.8,
            capsize=4.5,
            capthick=1.8,
            markersize=7.0,
            markerfacecolor=marker_color,
            markeredgecolor=marker_color,
            markeredgewidth=1.2,
            zorder=4,
        )

        # Titles and ticks with compact padding
        ax.set_title(panel["title"], fontsize=12, fontweight="bold", pad=6)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(panel["tick_labels"], fontsize=9.5)
        ax.set_xlim(-0.5, 4.5)

        # Y-axis range 0 to 100
        ax.set_ylim(-2, 102)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

        # Larger, bold R^2 text in blue in upper left corner
        r2_str = r"$\mathbf{R^2 = " + f"{r2:.{r2_decimals}f}" + r"}$"
        ax.text(
            0.06,
            0.92,
            r2_str,
            transform=ax.transAxes,
            color=line_color,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left",
        )

        # Slightly darker, clean dotted grid lines
        ax.grid(True, linestyle=":", color="#999999", alpha=0.7, linewidth=0.8)
        ax.set_axisbelow(True)

    # Y-axis ticks and labels: shown only on the leftmost graph with comfortable padding
    axes[0].set_ylabel("Perceived difference rating", fontsize=11.5, fontweight="bold", labelpad=6)
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)

    # Common X-axis label centered directly under middle subplot with comfortable padding
    axes[1].set_xlabel("Modulation amount", fontsize=12, fontweight="bold", labelpad=6)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        print(f"Figure successfully saved to: {output_path}")

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
        description="Generate 3-panel figure of human ratings across modulation conditions."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/listening_test_responses_postprocessed.tsv",
        help="Path to postprocessed human responses TSV (default: data/listening_test_responses_postprocessed.tsv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="../../out/figure_human_ratings.png",
        help="Path to save figure image (default: figures/figure_human_ratings.png). Supports .png, .pdf, .svg, etc.",
    )
    parser.add_argument(
        "--gap",
        "--wspace",
        type=float,
        default=0.10,
        dest="gap",
        help="White space gap between the three subplots in inches (default: 0.10).",
    )
    parser.add_argument(
        "--plot-size",
        type=float,
        default=2.85,
        dest="plot_size",
        help="Size of each square subplot in inches (default: 2.85).",
    )
    parser.add_argument(
        "--r2-decimals",
        type=int,
        default=2,
        help="Number of decimal places for R^2 value annotation (default: 2).",
    )
    parser.add_argument(
        "--fit-target",
        choices=["means", "all"],
        default="means",
        help="Target data points for linear regression line and R^2: 'means' (default) or 'all' trial ratings.",
    )
    parser.add_argument(
        "--line-color",
        default="#2a78d6",
        help="Color of the linear fit line and R^2 text (default: #2a78d6).",
    )
    parser.add_argument(
        "--marker-color",
        default="#111111",
        help="Color of markers and error bars (default: #111111).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in dots per inch for raster export (default: 300).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plot interactively in a window (only save to file).",
    )
    args = parser.parse_args()

    input_path = resolve_file_path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Error: Human ratings data file not found at: {input_path}\n")
        sys.exit(1)

    df_human = load_human_ratings(input_path)

    out_path = Path(args.output).expanduser().resolve() if args.output else None

    plot_human_ratings(
        df=df_human,
        output_path=out_path,
        plot_size=args.plot_size,
        gap=args.gap,
        line_color=args.line_color,
        marker_color=args.marker_color,
        r2_decimals=args.r2_decimals,
        fit_target=args.fit_target,
        dpi=args.dpi,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
