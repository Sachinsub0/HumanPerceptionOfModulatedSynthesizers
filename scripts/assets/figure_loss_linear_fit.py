"""Generate publication-ready figure of loss function linear fits across modulations.

Plots the 3-panel layout (Amplitude, Frequency (Hz), Irregularity (%)) for loss functions
arranged one per row. Subplots maintain an exact 1:1 aspect ratio. X-axis tick labels
and the shared 'Modulation amount' label appear only once at the very bottom row,
with column titles displayed at the top. Reference points are marked with squares,
subsequent points with circles.

Plotting options (--plot and --loss-fn):
- 'losses' (default): individual loss functions (or with Human Listeners as row 0 if --human-data is given).
- 'groups': 3 rows of aggregated representation groups (STFT, Wavelet, Neural).
- 'all': combines all individual loss functions and the 3 aggregated groups.
- --loss-fn / -l: filter and reorder analysis to specific loss function(s) (e.g. -l mfcc mss_log_lin).

Human benchmark option (--human-data / --human):
- When provided (e.g. data/listening_test_responses_postprocessed.tsv), participant ratings
  are preserved across all participants to compute empirical range, std, and CI before averaging,
  graphed as the first row ('Human Listeners') with y-axis ticks [0, 25, 50, 75, 100].
  The vertical gap between the human row and subsequent rows is configurable via --gap-human.

Representation group boundaries & dashed lines:
- A configurable vertical gap (--gap-group, default: 0.32) and subtle dashed horizontal
  separator line are drawn between distinct representation groups (Wavelet, Neural, STFT).

Normalization options (--normalize):
- '1' or 'max100': Peak/max normalization to [0, 100] globally per loss function.
- 'mean_max' or 'cond_max': Normalization by the maximum condition mean to [0, 100] (resists single-trial outliers).
- 'p95' or 'percentile95': Normalization by the 95th percentile to [0, 100].
- '2' or 'std': Zero-anchored standard deviation scaling (d / sigma).
- 'none' (default): Raw unnormalized distances.

Fit & Error Bar options:
- Linear line of best fit & R^2: toggleable via --no-fit / --hide-fit (default: shown).
- Anchor linear fit to y=0 at first x value: --anchor-zero / --anchor-first (default: anchored).
- Connecting dots: --connect-dots / --connect-points (straight black line segments).
- Error bars (mutually exclusive metric):
  - 95% Confidence Intervals: --ci or --error-bars ci
  - Standard Deviation: --std or --error-bars std (default)
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
    ("mss_log_lin", "MSS Log + Linear", "STFT"),
    ("mss_rev", "MSS Revisited", "STFT"),
    ("mfcc", "MFCC", "STFT"),
    # Wavelet Group
    ("scat1d_log1p", "Scat1D", "Wavelet"),
    ("jtfs_log1p", "JTFS", "Wavelet"),
    # Neural Group
    ("vggish", "VGGish", "Neural"),
    ("encodec48k", "EnCodec 48 kHz", "Neural"),
    ("clap2", "MS-CLAP", "Neural"),
    ("panns_wavegram_logmel", "PANNs WGLM", "Neural"),
]

# Aggregated representation groups and constituent loss functions
AGGREGATED_GROUPS = [
    {"key": "group_stft", "label": "STFT Group", "group": "STFT", "models": ["mss_log_lin", "mss_rev", "mfcc"]},
    {"key": "group_wavelet", "label": "Wavelet Group", "group": "Wavelet", "models": ["scat1d_log1p", "jtfs_log1p"]},
    {
        "key": "group_neural",
        "label": "Neural Group",
        "group": "Neural",
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
        "tick_labels": ["0.25", "0.5", "1", "2", "4"],
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


def load_human_data(data_path: Path) -> pd.DataFrame:
    """Load postprocessed human listening study responses as a benchmark.

    Retains all participant responses so empirical range, standard deviation,
    and confidence intervals are computed from the full participant distribution
    before being averaged into condition means.
    """
    sep = "\t" if data_path.suffix in [".tsv", ".txt"] else ","
    df_human = pd.read_csv(data_path, sep=sep)

    split_cols = df_human["trial_id"].str.split("_", expand=True)
    df_human["mod_type"] = split_cols[0]
    df_human["feature"] = split_cols[1]
    df_human["source"] = split_cols[2]
    df_human["distance"] = df_human["rating_score"].astype(float)
    df_human["loss_fn"] = "human"

    amount_map = {
        "amp": {
            "reference": 0.1,
            "condition_a": 0.3,
            "condition_b": 0.5,
            "condition_c": 0.7,
            "condition_d": 0.9,
        },
        "freq": {
            "reference": 0.25,
            "condition_a": 0.5,
            "condition_b": 1.0,
            "condition_c": 2.0,
            "condition_d": 4.0,
        },
        "reg": {
            "reference": 0.0,
            "condition_a": 0.125,
            "condition_b": 0.25,
            "condition_c": 0.375,
            "condition_d": 0.5,
        },
    }

    df_human["amount"] = [
        amount_map[m][s] for m, s in zip(df_human["mod_type"], df_human["rating_stimulus"])
    ]
    df_human["is_reference"] = df_human["rating_stimulus"] == "reference"

    return df_human


def normalize_distances(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Normalize distance values per loss function.

    Methods:
    - '1' or 'max100': Divide each loss function by its global max and scale to [0, 100].
    - 'mean_max' or 'cond_max': Divide each loss function by the maximum condition mean and scale to [0, 100] (resists single-trial outliers).
    - 'p95' or 'percentile95': Divide by the 95th percentile and scale to [0, 100].
    - '2' or 'std': Divide each loss function by its standard deviation around zero.
    - 'none': Keep raw distances unchanged.
    """
    method = str(method).lower().strip()
    if method in ("none", "raw", "0", ""):
        return df

    df_out = df.copy()
    for loss_key in df_out["loss_fn"].unique():
        if loss_key == "human":
            # Human listening study ratings are natively on [0, 100] scale
            continue

        mask = df_out["loss_fn"] == loss_key
        sub = df_out[mask]
        vals = sub["distance"].values

        if method in ("1", "max100", "max", "peak"):
            max_val = np.nanmax(vals)
            if max_val > 0:
                df_out.loc[mask, "distance"] = (vals / max_val) * 100.0
        elif method in ("mean_max", "cond_max"):
            cond_means = sub.groupby(["mod_type", "amount"])["distance"].mean()
            scale = cond_means.max()
            if scale > 0:
                df_out.loc[mask, "distance"] = (vals / scale) * 100.0
        elif method in ("p95", "percentile95"):
            scale = np.nanpercentile(vals, 95)
            if scale > 0:
                df_out.loc[mask, "distance"] = (vals / scale) * 100.0
        elif method in ("2", "std", "z-std", "sigma"):
            std_val = np.nanstd(vals)
            if std_val > 0:
                df_out.loc[mask, "distance"] = vals / std_val
        else:
            raise ValueError(
                f"Unknown normalization method: {method}. Choose '1' (max100), 'mean_max', 'p95', '2' (std), or 'none'."
            )

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


def compute_nice_5_ticks(v_max: float) -> tuple[list[float | int], float, float]:
    """Compute 5 clean, evenly spaced tickmarks [0, step, 2*step, 3*step, 4*step] for a given max value.

    Returns (ticks, y_lower, y_upper).
    """
    if v_max <= 0:
        return [0, 1, 2, 3, 4], -0.16, 4.16

    raw_step = v_max / 4.0
    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / (10 ** exponent)

    # Standard 1-2-5 and intermediate round step multipliers
    nice_steps = [1.0, 1.2, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    step_mult = min(s for s in nice_steps if s >= fraction - 1e-9)
    step = step_mult * (10 ** exponent)

    ticks = [round(i * step, 10) for i in range(5)]
    y_upper = ticks[-1] * 1.04
    y_lower = -0.04 * ticks[-1]

    # Convert to int if all values are exact integers
    if all(abs(t - round(t)) < 1e-8 for t in ticks):
        ticks = [int(round(t)) for t in ticks]

    return ticks, y_lower, y_upper


def normalize_loss_fns(raw_loss_fns: Sequence[str] | None) -> list[str] | None:
    """Clean and tokenize user-supplied loss function filter list."""
    if not raw_loss_fns:
        return None
    cleaned: list[str] = []
    for item in raw_loss_fns:
        for term in item.replace(",", " ").split():
            c = term.strip()
            if c and c not in cleaned:
                cleaned.append(c)
    return cleaned if cleaned else None


def prepare_plot_items(
    plot_mode: str,
    df_norm: pd.DataFrame,
    loss_fns: Sequence[str] | None = None,
) -> list[dict]:
    """Prepare row data specifications according to the chosen plot mode and loss function filter."""
    items: list[dict] = []

    # If human benchmark data is present, graph it as the first row
    if "human" in df_norm["loss_fn"].unique():
        items.append({
            "type": "human",
            "group": "human",
            "label": "Human Listeners",
            "sub_df": df_norm[df_norm["loss_fn"] == "human"],
        })

    # Lookup mapping for canonical display labels and groups
    loss_label_map = {key: label for key, label, _ in INDIVIDUAL_LOSS_FUNCTIONS}
    loss_group_map = {key: grp for key, _, grp in INDIVIDUAL_LOSS_FUNCTIONS}

    if plot_mode in ("losses", "all"):
        if loss_fns is not None:
            # Respect user-specified selection and ordering of loss functions
            for key in loss_fns:
                sub = df_norm[df_norm["loss_fn"] == key]
                if not sub.empty:
                    label = loss_label_map.get(key, key)
                    group = loss_group_map.get(key, "Other")
                    items.append({
                        "type": "individual",
                        "group": group,
                        "label": label,
                        "sub_df": sub,
                    })
        else:
            for key, label, grp in INDIVIDUAL_LOSS_FUNCTIONS:
                sub = df_norm[df_norm["loss_fn"] == key]
                if not sub.empty:
                    items.append({
                        "type": "individual",
                        "group": grp,
                        "label": label,
                        "sub_df": sub,
                    })

    if plot_mode in ("groups", "all"):
        for grp in AGGREGATED_GROUPS:
            group_models = grp["models"]
            if loss_fns is not None:
                group_models = [m for m in group_models if m in loss_fns]
            if group_models:
                sub = df_norm[df_norm["loss_fn"].isin(group_models)]
                if not sub.empty:
                    group_id = f"Aggregated_{grp['group']}" if plot_mode == "all" else grp["group"]
                    items.append({
                        "type": "group",
                        "group": group_id,
                        "label": grp["label"],
                        "sub_df": sub,
                    })

    return items


def plot_loss_linear_fits(
    df: pd.DataFrame,
    plot_mode: str = "losses",
    loss_fns: Sequence[str] | None = None,
    normalize: str = "none",
    show_fit: bool = True,
    anchor_zero: bool = True,
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
    gap_human: float = 0.32,
    gap_group: float = 0.32,
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
    loss_fns_clean = normalize_loss_fns(loss_fns)
    row_items = prepare_plot_items(plot_mode, df_plot, loss_fns=loss_fns_clean)

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

    # Determine vertical gaps between consecutive rows: row_gaps[r] is between row r (above) and r+1 (below)
    row_gaps: list[float] = []
    has_separator: list[bool] = []
    for r in range(n_rows - 1):
        curr_item = row_items[r]
        next_item = row_items[r + 1]
        if curr_item.get("type") == "human":
            row_gaps.append(gap_human)
            has_separator.append(True)
        elif curr_item.get("group") != next_item.get("group"):
            row_gaps.append(gap_group)
            has_separator.append(True)
        else:
            row_gaps.append(gap_y)
            has_separator.append(False)

    fig_w = margin_left + n_cols * plot_size + (n_cols - 1) * gap_x + margin_right
    total_gaps_y = sum(row_gaps)
    fig_h = margin_bottom + n_rows * plot_size + total_gaps_y + margin_top

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # Compute bottom coordinate in inches for each row r (r = 0 is topmost, r = n_rows - 1 is bottommost)
    b_in_list = [0.0] * n_rows
    b_in_list[n_rows - 1] = margin_bottom
    for r in range(n_rows - 2, -1, -1):
        b_in_list[r] = b_in_list[r + 1] + plot_size + row_gaps[r]

    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        b_in = b_in_list[r]
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
        is_human = item.get("type") == "human"

        # Calculate y-limits and ticks (5 tickmarks for human data and all loss functions)
        if is_human or norm_clean in ("1", "max100", "max", "peak", "mean_max", "cond_max", "p95"):
            y_upper = 104.0
            y_lower = -4.0
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

            y_ticks, y_lower, y_upper = compute_nice_5_ticks(row_max)

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
                if anchor_zero:
                    x0 = float(x_indices[0])
                    u = x_indices - x0
                    denom = float(np.sum(u ** 2))
                    slope = float(np.sum(u * means_arr) / denom) if denom > 0 else 0.0
                    y_pred = slope * u
                    ss_res = float(np.sum((means_arr - y_pred) ** 2))
                    ss_tot = float(np.sum((means_arr - np.mean(means_arr)) ** 2))
                    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

                    x_line = np.linspace(x0, 4.25, 100)
                    y_line = slope * (x_line - x0)
                else:
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

            # Y-axis ticks: exactly 5 tickmarks for human data and all loss functions
            ax.set_yticks(y_ticks)

            # Column titles: only on the very top row (r_idx == 0)
            if r_idx == 0:
                ax.set_title(col["title"], fontsize=11.5, fontweight="bold", pad=6)

            # X-ticks: set explicitly on ALL rows so vertical gridlines are drawn at every point
            ax.set_xticks(x_indices)
            if r_idx == n_rows - 1:
                ax.set_xticklabels(col["tick_labels"], fontsize=9)
            else:
                ax.tick_params(labelbottom=False, bottom=False)

            # Y-axis label and ticks: only on the leftmost column (c_idx == 0)
            if c_idx == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight="bold", labelpad=5)
                ax.tick_params(labelsize=8.5)
            else:
                ax.tick_params(labelleft=False, left=False)

            # Dotted gridlines across all subplots
            ax.grid(True, which="major", linestyle=":", color="#999999", alpha=0.7, linewidth=0.7)
            ax.set_axisbelow(True)

    # Subtle dashed separator lines centered between Human benchmark and between representation groups
    for r in range(n_rows - 1):
        if has_separator[r]:
            sep_y_in = (b_in_list[r] + b_in_list[r + 1] + plot_size) / 2.0
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


def resolve_output_path(path_str: str) -> Path:
    """Resolve output file path properly relative to cwd, repo root, or script location."""
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parent.parent.parent
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
        description="Generate figure of loss function linear fits across modulations (losses, groups, or all)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/distances.tsv",
        help="Path to distances TSV/CSV dataset (default: data/distances.tsv)",
    )
    parser.add_argument(
        "--human-data",
        "--human",
        nargs="?",
        const="data/listening_test_responses_postprocessed.tsv",
        default="data/listening_test_responses_postprocessed.tsv",
        # default=None,
        dest="human_data",
        help="Optional path to postprocessed human listening responses TSV (default when flag used without argument: data/listening_test_responses_postprocessed.tsv). When provided, human ratings are graphed as the first row.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="../../out/figure_loss_linear_fit.pdf",
        help="Path to save figure image (default: figures/figure_loss_linear_fit.png). Supports .png, .pdf, .svg, etc.",
    )
    parser.add_argument(
        "--plot",
        choices=["losses", "groups", "all"],
        default="losses",
        help="Plotting mode: 'losses' (individual loss function rows), 'groups' (3 aggregated rows: STFT, Wavelet, Neural), or 'all'. Default: losses.",
    )
    parser.add_argument(
        "--loss-fn",
        "--loss-fns",
        "-l",
        nargs="+",
        default=None,
        # default=["mss_log_lin", "mss_rev", "mfcc", "scat1d_log1p", "jtfs_log1p"],
        # default=["vggish", "encodec48k", "clap2", "panns_wavegram_logmel"],
        help="Filter analysis to specific loss function(s) (e.g. -l mfcc mss_log_lin)",
    )
    parser.add_argument(
        "--normalize",
        choices=["1", "max100", "mean_max", "cond_max", "p95", "2", "std", "none"],
        default="none",
        help="Distance normalization method: '1' or 'max100' (Peak scale to [0, 100]), 'mean_max' (scale by maximum condition mean), 'p95' (scale by 95th percentile), '2' or 'std' (scale by standard deviation), or 'none'.",
    )
    parser.add_argument(
        "--no-fit",
        "--hide-fit",
        action="store_false",
        dest="show_fit",
        default=True,
        help="Do not display linear line of best fit and R^2 value annotation.",
    )
    parser.add_argument(
        "--anchor-zero",
        "--anchor-origin",
        "--anchor-first",
        "--anchor",
        action="store_true",
        dest="anchor_zero",
        default=True,
        help="Anchor the linear line of best fit to pass through y=0 at the first x value.",
    )
    parser.add_argument(
        "--connect-dots",
        "--connect-points",
        action="store_true",
        dest="connect_dots",
        default=False,
        help="Connect data points with straight black line segments.",
    )

    # Mutually exclusive error bar metric options
    err_group = parser.add_mutually_exclusive_group()
    err_group.add_argument(
        "--error-bars",
        choices=["ci", "std", "none"],
        dest="error_bars",
        default="std",
        help="Error bar metric: 'ci' (95%% confidence intervals), 'std' (sample standard deviation, default), or 'none' (no error bars).",
    )

    # Shaded region / error style options
    parser.add_argument(
        "--error-style",
        choices=["bars", "shade", "both"],
        default="bars",
        dest="error_style",
        help="Error visual style: 'bars' (default: vertical error bars with caps), 'shade' (shaded ribbon), or 'both'.",
    )
    parser.add_argument(
        "--shade-range",
        "--range-shade",
        "--min-max",
        action="store_true",
        dest="shade_range",
        default=True,
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
        help="Vertical gap between model rows in inches (default: 0.08).",
    )
    parser.add_argument(
        "--gap-human",
        "--human-gap",
        type=float,
        default=0.32,
        dest="gap_human",
        help="Vertical gap between the human benchmark row and subsequent model rows in inches (default: 0.32).",
    )
    parser.add_argument(
        "--gap-group",
        "--group-gap",
        type=float,
        default=0.32,
        dest="gap_group",
        help="Vertical gap between representation groups in inches (default: 0.32).",
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

    # Load human listening study data if provided
    if args.human_data:
        human_path = resolve_file_path(args.human_data)
        if not human_path.exists():
            sys.stderr.write(f"Warning: Human data file not found at: {human_path}. Proceeding without human data.\n")
        else:
            df_human = load_human_data(human_path)
            df_dist = pd.concat([df_human, df_dist], ignore_index=True)

    out_path = resolve_output_path(args.output) if args.output else None

    plot_loss_linear_fits(
        df=df_dist,
        plot_mode=args.plot,
        loss_fns=args.loss_fn,
        normalize=args.normalize,
        show_fit=args.show_fit,
        anchor_zero=args.anchor_zero,
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
        gap_human=args.gap_human,
        gap_group=args.gap_group,
        line_color=args.line_color,
        marker_color=args.marker_color,
        r2_decimals=args.r2_decimals,
        dpi=args.dpi,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
