"""Generate publication-ready figure of loss function linear fits across modulations.

Plots the 3-panel layout (Amplitude, Frequency (Hz), Irregularity (%)) for loss functions
arranged one per row. Subplots maintain an exact 1:1 aspect ratio. X-axis tick labels
and the shared 'Modulation amount' label appear only once at the very bottom row,
with column titles displayed at the top. Reference points are marked with squares,
subsequent points with circles.

Plotting options (--plot):
- 'losses' (default): 9 rows of individual loss functions.
- 'groups': 3 rows of aggregated representation groups (STFT, Wavelet, Neural).
- 'all': 12 rows combining all 9 individual loss functions and the 3 aggregated groups.

Normalization options (--normalize):
- '1' or 'max100' (default): Peak/max normalization to [0, 100] globally per loss function.
- '2' or 'std': Zero-anchored standard deviation scaling (d / sigma).
- 'none': Raw unnormalized distances.

Fit & Error Bar options:
- Linear line of best fit & R^2: toggleable via --no-fit / --hide-fit (default: shown).
- Connecting dots: --connect-dots / --connect-points (straight black line segments).
- Error bars (mutually exclusive metric):
  - 95% Confidence Intervals: --ci or --error-bars ci (default)
  - Standard Deviation: --std or --error-bars std
  - No error bars: --no-ci, --no-error-bars, or --error-bars none
- Error representation style:
  - --shade or --error-style shade: plot CI or STD as a semi-transparent shaded region.
  - --error-style bars (default): plot vertical error bars with caps.
  - --error-style both: plot both shaded region and vertical error bars.
  - --shade-range: also shade the [min, max] range with a lighter color.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress

# Canonical individual loss functions
INDIVIDUAL_LOSS_FUNCTIONS = [
    # STFT Group
    ("mss_log_lin", "MSS Log + Lin.", "STFT"),
    ("mss_rev", "MSS Revisited", "STFT"),
    ("mfcc", "MFCC", "STFT"),
    # Wavelet Group
    ("scat1d_log1p", "Scat1D", "Wavelet"),
    ("jtfs_log1p", "JTFS", "Wavelet"),
    # Neural Group
    ("vggish", "VGGish", "Neural"),
    ("encodec48k", "EnCodec 48~kHz", "Neural"),
    ("clap2", "MS-CLAP", "Neural"),
    ("panns_wavegram_logmel", "PANNs WGLM", "Neural"),
]

# Aggregated representation groups and constituent loss functions
AGGREGATED_GROUPS = [
    {
        "key": "group_stft",
        "label": "STFT Group",
        "models": ["mss_log_lin", "mss_rev", "mfcc"],
    },
    {
        "key": "group_wavelet",
        "label": "Wavelet Group",
        "models": ["scat1d_log1p", "jtfs_log1p"],
    },
    {
        "key": "group_neural",
        "label": "Neural Group",
        "models": ["vggish", "encodec48k", "clap2", "panns_wavegram_logmel"],
    },
]

# Modulation conditions and column configurations
MODULATION_COLUMNS = [
    {
        "key": "amp",
        "title": "Amplitude",
        "tick_labels": ["0.1", "0.3", "0.5", "0.7", "0.9"],
    },
    {
        "key": "freq",
        "title": "Frequency (Hz)",
        "tick_labels": ["0.25", "0.50", "1.00", "2.00", "4.00"],
    },
    {
        "key": "reg",
        "title": "Irregularity (%)",
        "tick_labels": ["0", "12.5", "25", "37.5", "50"],
    },
]


def load_distances_data(data_path: Path) -> pd.DataFrame:
    """Load distances dataset."""
    sep = "\t" if data_path.suffix in [".tsv", ".txt"] else ","
    return pd.read_csv(data_path, sep=sep)


def normalize_distances(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Normalize distance values per loss function.

    Methods:
    - '1' or 'max100': Divide each loss function by its global max and scale to [0, 100].
    - '2' or 'std': Divide each loss function by its standard deviation around zero.
    - 'none': Keep raw distances unchanged.
    """
    method = str(method).lower().strip()
    if method in ("none", "raw", "0", ""):
        return df

    df_out = df.copy()
    for loss_key in df_out["loss_fn"].unique():
        mask = df_out["loss_fn"] == loss_key
        vals = df_out.loc[mask, "distance"].values
        if method in ("1", "max100", "max", "peak"):
            max_val = np.nanmax(vals)
            if max_val > 0:
                df_out.loc[mask, "distance"] = (vals / max_val) * 100.0
        elif method in ("2", "std", "z-std", "sigma"):
            std_val = np.nanstd(vals)
            if std_val > 0:
                df_out.loc[mask, "distance"] = vals / std_val
        else:
            raise ValueError(f"Unknown normalization method: {method}. Choose '1' (max100), '2' (std), or 'none'.")

    return df_out


def compute_point_stats(
    values: np.ndarray,
    error_mode: str = "ci",
    ci_level: float = 0.95,
) -> tuple[float, Optional[float]]:
    """Compute mean and error bar half-width (CI, STD, or None)."""
    valid = values[~np.isnan(values)]
    n = len(valid)
    if n == 0:
        return 0.0, None if error_mode == "none" else 0.0
    mean_val = float(np.mean(valid))

    if error_mode in ("none", "no", "off"):
        return mean_val, None
    if n <= 1:
        return mean_val, 0.0

    s = float(np.std(valid, ddof=1))
    if error_mode == "std":
        return mean_val, s
    elif error_mode == "ci":
        if s == 0.0:
            return mean_val, 0.0
        se = s / np.sqrt(n)
        t_crit = float(stats.t.ppf((1.0 + ci_level) / 2.0, df=n - 1))
        return mean_val, t_crit * se
    else:
        raise ValueError(f"Unknown error_mode: {error_mode}. Choose 'ci', 'std', or 'none'.")


def prepare_plot_items(
    plot_mode: str,
    df_norm: pd.DataFrame,
) -> list[dict]:
    """Prepare row data specifications according to the chosen plot mode."""
    items: list[dict] = []

    if plot_mode in ("losses", "all"):
        for key, label, _ in INDIVIDUAL_LOSS_FUNCTIONS:
            items.append({
                "type": "individual",
                "label": label,
                "sub_df": df_norm[df_norm["loss_fn"] == key],
            })

    if plot_mode in ("groups", "all"):
        for grp in AGGREGATED_GROUPS:
            items.append({
                "type": "group",
                "label": grp["label"],
                "sub_df": df_norm[df_norm["loss_fn"].isin(grp["models"])],
            })

    return items


def plot_loss_linear_fits(
    df: pd.DataFrame,
    plot_mode: str = "losses",
    normalize: str = "1",
    show_fit: bool = True,
    connect_dots: bool = False,
    error_mode: str = "ci",
    error_style: str = "bars",
    shade_range: bool = False,
    range_alpha: float = 0.07,
    shade_alpha: float = 0.16,
    ci_level: float = 0.95,
    output_path: Optional[Path] = None,
    plot_size: float = 2.0,
    gap_x: float = 0.08,
    gap_y: float = 0.08,
    line_color: str = "#2a78d6",
    marker_color: str = "#111111",
    r2_decimals: int = 2,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """Generate publication-ready figure of linear fits for loss functions and/or groups."""
    norm_clean = str(normalize).lower().strip()
    if plot_mode in ("groups", "all") and norm_clean in ("none", "raw", "0", ""):
        print("Note: Aggregating groups requires distance normalization. Defaulting to Option 1 (max100).")
        norm_clean = "1"

    df_plot = normalize_distances(df, norm_clean)
    row_items = prepare_plot_items(plot_mode, df_plot)

    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.85,
    })

    n_rows = len(row_items)
    n_cols = len(MODULATION_COLUMNS)

    margin_left = 0.85
    margin_right = 0.06
    margin_bottom = 0.44
    margin_top = 0.32

    fig_w = margin_left + n_cols * plot_size + (n_cols - 1) * gap_x + margin_right
    fig_h = margin_bottom + n_rows * plot_size + (n_rows - 1) * gap_y + margin_top

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        row_from_bottom = n_rows - 1 - r
        b_in = margin_bottom + row_from_bottom * (plot_size + gap_y)
        for c in range(n_cols):
            l_in = margin_left + c * (plot_size + gap_x)
            ax = fig.add_axes([
                l_in / fig_w,
                b_in / fig_h,
                plot_size / fig_w,
                plot_size / fig_h,
            ])
            axes[r, c] = ax

    x_indices = np.arange(5)

    for r_idx, item in enumerate(row_items):
        sub_df = item["sub_df"]
        row_label = item["label"]

        # Calculate shared y-upper limit for this row across all 3 conditions
        if norm_clean in ("1", "max100", "max", "peak"):
            y_upper = 104.0
            y_lower = -2.0
            y_ticks = [0, 25, 50, 75, 100]
        else:
            row_max = 0.0
            for col in MODULATION_COLUMNS:
                sub_mod = sub_df[sub_df["mod_type"] == col["key"]]
                amounts = sorted(sub_mod["amount"].unique())
                for a in amounts:
                    vals = sub_mod[sub_mod["amount"] == a]["distance"].values
                    m, err = compute_point_stats(vals, error_mode=error_mode, ci_level=ci_level)
                    val = m + (err if err is not None else 0.0)
                    if shade_range and len(vals) > 0:
                        val = max(val, float(np.nanmax(vals)))
                    if val > row_max:
                        row_max = val

            y_upper = row_max * 1.15 if row_max > 0 else 1.0
            y_lower = -0.02 * y_upper
            y_ticks = None

        for c_idx, col in enumerate(MODULATION_COLUMNS):
            ax = axes[r_idx, c_idx]
            sub_mod = sub_df[sub_df["mod_type"] == col["key"]]
            amounts = sorted(sub_mod["amount"].unique())

            means: list[float] = []
            errs: list[Optional[float]] = []
            mins: list[float] = []
            maxs: list[float] = []
            for a in amounts:
                vals = sub_mod[sub_mod["amount"] == a]["distance"].values
                m, err = compute_point_stats(vals, error_mode=error_mode, ci_level=ci_level)
                means.append(m)
                errs.append(err)
                valid_vals = vals[~np.isnan(vals)]
                if len(valid_vals) > 0:
                    mins.append(float(np.min(valid_vals)))
                    maxs.append(float(np.max(valid_vals)))
                else:
                    mins.append(m)
                    maxs.append(m)

            means_arr = np.array(means)
            mins_arr = np.array(mins)
            maxs_arr = np.array(maxs)
            has_errs = error_mode != "none" and any(e is not None for e in errs)
            yerr_arr = np.array([e if e is not None else 0.0 for e in errs]) if has_errs else None

            # 0. Shaded region for min and max range (lighter color)
            if shade_range and len(mins_arr) == len(x_indices):
                ax.fill_between(
                    x_indices,
                    mins_arr,
                    maxs_arr,
                    color=marker_color,
                    alpha=range_alpha,
                    edgecolor="none",
                    zorder=1.5,
                )

            # 1. Shaded region for CI or STD
            if error_style in ("shade", "both") and has_errs and yerr_arr is not None:
                y_lower_band = np.maximum(y_lower, means_arr - yerr_arr)
                y_upper_band = means_arr + yerr_arr
                ax.fill_between(
                    x_indices,
                    y_lower_band,
                    y_upper_band,
                    color=marker_color,
                    alpha=shade_alpha,
                    edgecolor="none",
                    zorder=2,
                )

            # 2. Linear fit line and R^2
            if show_fit:
                reg = linregress(x_indices, means_arr)
                r2 = reg.rvalue ** 2

                x_line = np.linspace(-0.25, 4.25, 100)
                y_line = reg.slope * x_line + reg.intercept
                ax.plot(
                    x_line,
                    y_line,
                    color=line_color,
                    linewidth=2.0,
                    linestyle="-",
                    zorder=2.5,
                )

                # R^2 text in upper left corner
                r2_str = r"$\mathbf{R^2 = " + f"{r2:.{r2_decimals}f}" + r"}$"
                ax.text(
                    0.06,
                    0.92,
                    r2_str,
                    transform=ax.transAxes,
                    color=line_color,
                    fontsize=10.5,
                    fontweight="bold",
                    va="top",
                    ha="left",
                )

            # 3. Straight black line segments connecting the dots
            if connect_dots:
                ax.plot(
                    x_indices,
                    means_arr,
                    color=marker_color,
                    linewidth=1.5,
                    linestyle="-",
                    zorder=3,
                )

            # 4. Markers and vertical error bars
            show_vertical_bars = error_style in ("bars", "both") and has_errs
            ref_err = yerr_arr[0] if (show_vertical_bars and yerr_arr is not None) else None
            rem_err = yerr_arr[1:] if (show_vertical_bars and yerr_arr is not None) else None

            ax.errorbar(
                x_indices[0],
                means_arr[0],
                yerr=ref_err,
                fmt="s",
                color=marker_color,
                ecolor=marker_color,
                elinewidth=1.5,
                capsize=3.5 if show_vertical_bars else 0,
                capthick=1.5,
                markersize=5.5,
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                markeredgewidth=1.1,
                zorder=4,
            )
            ax.errorbar(
                x_indices[1:],
                means_arr[1:],
                yerr=rem_err,
                fmt="o",
                color=marker_color,
                ecolor=marker_color,
                elinewidth=1.5,
                capsize=3.5 if show_vertical_bars else 0,
                capthick=1.5,
                markersize=5.5,
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                markeredgewidth=1.1,
                zorder=4,
            )

            # Subplot aspect ratio 1:1
            ax.set_box_aspect(1)
            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(y_lower, y_upper)

            # Y-axis ticks
            if y_ticks is not None:
                ax.set_yticks(y_ticks)
            else:
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune=None))

            # Column titles: only on the very top row (r_idx == 0)
            if r_idx == 0:
                ax.set_title(col["title"], fontsize=11.5, fontweight="bold", pad=6)

            # X-ticks: only on the very bottom row (r_idx == n_rows - 1)
            if r_idx == n_rows - 1:
                ax.set_xticks(x_indices)
                ax.set_xticklabels(col["tick_labels"], fontsize=9)
            else:
                ax.tick_params(labelbottom=False, bottom=False)

            # Y-axis label and ticks: only on the leftmost column (c_idx == 0)
            if c_idx == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight="bold", labelpad=5)
                ax.tick_params(labelsize=8.5)
            else:
                ax.tick_params(labelleft=False, left=False)

            # Dotted gridlines
            ax.grid(True, linestyle=":", color="#999999", alpha=0.7, linewidth=0.7)
            ax.set_axisbelow(True)

    # In 'all' mode: draw subtle dashed line separating individual models from aggregated groups
    if plot_mode == "all":
        sep_y_in = margin_bottom + 3 * (plot_size + gap_y) - (gap_y / 2.0)
        sep_y_norm = sep_y_in / fig_h
        fig.add_artist(
            plt.Line2D(
                [margin_left / fig_w, 1.0 - (margin_right / fig_w)],
                [sep_y_norm, sep_y_norm],
                color="#888888",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )
        )

    # Common X-axis label centered at the bottom
    axes[n_rows - 1, 1].set_xlabel("Modulation amount", fontsize=11.5, fontweight="bold", labelpad=5)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
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
        description="Generate figure of loss function linear fits across modulations (losses, groups, or all)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/distances.tsv",
        help="Path to distances TSV/CSV dataset (default: data/distances.tsv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="../../out/figure_loss_linear_fit.png",
        help="Path to save figure image (default: figures/figure_loss_linear_fit.png). Supports .png, .pdf, .svg, etc.",
    )
    parser.add_argument(
        "--plot",
        choices=["losses", "groups", "all"],
        default="losses",
        # default="groups",
        help="Plotting mode: 'losses' (9 individual rows), 'groups' (3 aggregated rows: STFT, Wavelet, Neural), or 'all' (12 rows). Default: losses.",
    )
    parser.add_argument(
        "--normalize",
        choices=["1", "max100", "2", "std", "none"],
        default="1",
        help="Distance normalization method: '1' or 'max100' (default: Peak scale to [0, 100]), '2' or 'std' (scale by standard deviation), or 'none'.",
    )
    parser.add_argument(
        "--no-fit",
        "--hide-fit",
        action="store_false",
        dest="show_fit",
        default=True,
        # default=False,
        help="Do not display linear line of best fit and R^2 value annotation.",
    )
    parser.add_argument(
        "--connect-dots",
        "--connect-points",
        action="store_true",
        dest="connect_dots",
        default=False,
        # default=True,
        help="Connect data points with straight black line segments.",
    )

    # Mutually exclusive error bar metric options
    err_group = parser.add_mutually_exclusive_group()
    err_group.add_argument(
        "--error-bars",
        choices=["ci", "std", "none"],
        dest="error_bars",
        default="std",
        help="Error bar metric: 'ci' (95%% confidence intervals, default), 'std' (sample standard deviation), or 'none' (no error bars).",
    )
    # err_group.add_argument(
    #     "--ci",
    #     action="store_const",
    #     dest="error_bars",
    #     const="ci",
    #     help="Display 95%% Student's t confidence intervals as error metric (default).",
    # )
    # err_group.add_argument(
    #     "--std",
    #     action="store_const",
    #     dest="error_bars",
    #     const="std",
    #     help="Display standard deviation as error metric (instead of confidence intervals).",
    # )
    # err_group.add_argument(
    #     "--no-ci",
    #     "--no-error-bars",
    #     action="store_const",
    #     dest="error_bars",
    #     const="none",
    #     help="Do not display any error bars or shaded error region.",
    # )

    # Shaded region / error style options
    parser.add_argument(
        "--error-style",
        choices=["bars", "shade", "both"],
        default="bars",
        # default="shade",
        dest="error_style",
        help="Error visual style: 'bars' (default: vertical error bars with caps), 'shade' (shaded ribbon), or 'both'.",
    )
    # parser.add_argument(
    #     "--shade",
    #     "--shaded",
    #     action="store_const",
    #     const="shade",
    #     dest="error_style",
    #     help="Plot the CI or STD as a semi-transparent shaded region instead of vertical error bars.",
    # )
    parser.add_argument(
        "--shade-range",
        "--range-shade",
        "--min-max",
        action="store_true",
        dest="shade_range",
        default=False,
        # default=True,
        help="Shade the full [min, max] observed range with a lighter color than the CI/STD shading.",
    )
    parser.add_argument(
        "--range-alpha",
        type=float,
        default=0.07,
        help="Opacity for min-max range shading (default: 0.07).",
    )
    parser.add_argument(
        "--shade-alpha",
        type=float,
        default=0.16,
        help="Opacity for CI or STD shading (default: 0.16).",
    )

    parser.add_argument(
        "--plot-size",
        type=float,
        default=2.0,
        dest="plot_size",
        help="Size of each square subplot in inches (default: 2.0).",
    )
    parser.add_argument(
        "--gap-x",
        type=float,
        default=0.08,
        dest="gap_x",
        help="Horizontal gap between subplots in inches (default: 0.08).",
    )
    parser.add_argument(
        "--gap-y",
        type=float,
        default=0.08,
        dest="gap_y",
        help="Vertical gap between rows in inches (default: 0.08).",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level for Student's t confidence intervals (default: 0.95).",
    )
    parser.add_argument(
        "--r2-decimals",
        type=int,
        default=2,
        help="Number of decimal places for R^2 value annotation (default: 2).",
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
        sys.stderr.write(f"Error: Distances data file not found at: {input_path}\n")
        sys.exit(1)

    df_dist = load_distances_data(input_path)

    out_path = Path(args.output).expanduser().resolve() if args.output else None

    plot_loss_linear_fits(
        df=df_dist,
        plot_mode=args.plot,
        normalize=args.normalize,
        show_fit=args.show_fit,
        connect_dots=args.connect_dots,
        error_mode=args.error_bars,
        error_style=args.error_style,
        shade_range=args.shade_range,
        range_alpha=args.range_alpha,
        shade_alpha=args.shade_alpha,
        ci_level=args.ci_level,
        output_path=out_path,
        plot_size=args.plot_size,
        gap_x=args.gap_x,
        gap_y=args.gap_y,
        line_color=args.line_color,
        marker_color=args.marker_color,
        r2_decimals=args.r2_decimals,
        dpi=args.dpi,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
