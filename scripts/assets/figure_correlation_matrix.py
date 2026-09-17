"""Generate publication-ready correlation matrix figure and terminal table for audio distance models.

Reads data/correlation_results.tsv and optionally data/noise_ceiling_results.tsv,
extracts group Spearman (rho) or Pearson (r) correlation values presented in
table_correlation_small.py, prints the formatted correlation matrix to the terminal,
and generates a publication-quality heatmap figure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

# Modulation conditions
CONDITIONS = ["amp", "freq", "reg"]

CONDITION_MAP = {
    "amp": "Amplitude",
    "freq": "Frequency",
    "reg": "Irregularity",
}

# Method groups matching table_correlation_small.py and figure_variance.py
METHOD_GROUPS = [
    [
        ("mss_log_lin", "MSS Log + Lin.", "STFT"),
        ("mss_rev", "MSS Revisited", "STFT"),
        ("mfcc", "MFCC", "STFT"),
    ],
    [
        ("scat1d_log1p", "Scat1D", "Wavelet"),
        ("jtfs_log1p", "JTFS", "Wavelet"),
    ],
    [
        ("vggish", "VGGish", "Neural"),
        ("encodec48k", "EnCodec 48 kHz", "Neural"),
        ("clap2", "MS-CLAP", "Neural"),
        ("panns_wavegram_logmel", "PANNs WGLM", "Neural"),
    ],
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


def format_sig_figs(val: float, precision: int = 3) -> str:
    """Format a float to a fixed number of significant figures, padding trailing zeros."""
    if val is None or pd.isna(val):
        return ""
    s = f"{val:.{precision}g}"
    if "e" in s or "E" in s:
        return f"{val:.{precision}f}"
    parts = s.split(".")
    if len(parts) == 1:
        needed = precision - len(parts[0])
        return parts[0] + "." + "0" * max(0, needed)
    sig_digits = (
        len(parts[1].lstrip("0"))
        if parts[0] == "0"
        else len(parts[0].lstrip("0")) + len(parts[1])
    )
    needed = precision - sig_digits
    if needed > 0:
        s += "0" * needed
    return s


def get_asterisks(p_val: float) -> str:
    """Return significance asterisks based on p-value."""
    if p_val is None or pd.isna(p_val):
        return ""
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    return ""


def load_correlation_matrix(
    corr_path: Path,
    nc_path: Optional[Path] = None,
    metric: str = "spearman",
    include_noise_ceiling: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str, str]]]:
    """Load correlation results and construct the Model x Condition correlation matrix.

    Args:
        corr_path: Path to correlation_results.tsv.
        nc_path: Optional path to noise_ceiling_results.tsv.
        metric: 'spearman' or 'pearson'.
        include_noise_ceiling: Whether to include the noise ceiling row.

    Returns:
        corr_matrix: DataFrame of shape (n_models, 3) with correlation coefficients.
        p_matrix: DataFrame of shape (n_models, 3) with p-values.
        models_meta: List of (model_key, display_name, group_name).
    """
    sep_corr = "\t" if corr_path.suffix in [".tsv", ".txt"] else ","
    df = pd.read_csv(corr_path, sep=sep_corr)

    if "loss_fn" in df.columns:
        df["canonical_loss"] = df["loss_fn"].map(lambda x: LOSS_FN_ALIASES.get(x, x))

    val_col = f"{metric}_group"
    p_col = f"{metric}_group_p"

    models_meta: list[tuple[str, str, str]] = []
    matrix_rows: list[dict[str, float]] = []
    p_rows: list[dict[str, float]] = []

    # Filter modulation granularity
    mod_df = df[df["granularity"] == "modulation"]

    # Optional noise ceiling row
    if include_noise_ceiling and nc_path and nc_path.exists():
        sep_nc = "\t" if nc_path.suffix in [".tsv", ".txt"] else ","
        nc_df = pd.read_csv(nc_path, sep=sep_nc)
        nc_mod = nc_df[nc_df["granularity"] == "modulation"]

        nc_vals = {}
        nc_p_vals = {}
        for cond in CONDITIONS:
            match = nc_mod[nc_mod["condition"] == cond]
            if not match.empty:
                nc_vals[CONDITION_MAP[cond]] = float(match.iloc[0][f"{metric}_group"])
                nc_p_vals[CONDITION_MAP[cond]] = 0.0  # reference benchmark
            else:
                nc_vals[CONDITION_MAP[cond]] = float("nan")
                nc_p_vals[CONDITION_MAP[cond]] = float("nan")

        models_meta.append(("noise_ceiling", "Noise Ceiling", "Benchmark"))
        matrix_rows.append(nc_vals)
        p_rows.append(nc_p_vals)

    for group in METHOD_GROUPS:
        for canon_key, display_name, grp_name in group:
            row_vals = {}
            row_p_vals = {}
            for cond in CONDITIONS:
                cond_df = mod_df[mod_df["condition"] == cond]
                match = cond_df[cond_df["canonical_loss"] == canon_key]
                if match.empty:
                    match = cond_df[cond_df["loss_fn"] == canon_key]

                col_name = CONDITION_MAP[cond]
                if not match.empty:
                    r = match.iloc[0]
                    row_vals[col_name] = float(r[val_col])
                    row_p_vals[col_name] = float(r[p_col])
                else:
                    row_vals[col_name] = float("nan")
                    row_p_vals[col_name] = float("nan")

            models_meta.append((canon_key, display_name, grp_name))
            matrix_rows.append(row_vals)
            p_rows.append(row_p_vals)

    row_names = [m[1] for m in models_meta]
    col_names = [CONDITION_MAP[c] for c in CONDITIONS]

    corr_matrix = pd.DataFrame(matrix_rows, index=row_names, columns=col_names)
    p_matrix = pd.DataFrame(p_rows, index=row_names, columns=col_names)

    return corr_matrix, p_matrix, models_meta


def print_correlation_matrix(
    corr_matrix: pd.DataFrame,
    p_matrix: pd.DataFrame,
    models_meta: list[tuple[str, str, str]],
    metric: str = "spearman",
    sig_digits: int = 3,
    output_format: str = "table",
) -> None:
    """Print the correlation matrix to terminal formatted nicely."""
    symbol = r"\rho" if metric == "spearman" else "r"
    metric_label = f"Group {metric.capitalize()} Correlation Matrix ({symbol})"

    if output_format == "csv":
        print(f"# {metric_label}")
        corr_matrix.to_csv(sys.stdout)
        return

    if output_format == "markdown":
        print(f"### {metric_label}\n")
        header = "| Method | Group | " + " | ".join(corr_matrix.columns) + " |"
        sep = (
            "| :--- | :--- | " + " | ".join([":---:"] * len(corr_matrix.columns)) + " |"
        )
        print(header)
        print(sep)
        for (_, display_name, grp), (_, r_vals), (_, r_p) in zip(
            models_meta, corr_matrix.iterrows(), p_matrix.iterrows()
        ):
            cells = []
            for col in corr_matrix.columns:
                val_str = format_sig_figs(r_vals[col], sig_digits)
                ast = get_asterisks(r_p[col])
                cells.append(f"{val_str}{ast}")
            print(f"| {display_name} | {grp} | " + " | ".join(cells) + " |")
        print()
        return

    # Standard aligned table format
    col_widths = [max(len(c), 14) for c in corr_matrix.columns]
    name_width = max(len("Method"), max(len(m[1]) for m in models_meta) + 2)
    group_width = max(len("Group"), max(len(m[2]) for m in models_meta) + 2)

    tot_width = name_width + group_width + sum(col_widths) + 4

    print("=" * tot_width)
    print(f"  {metric_label.upper()}")
    print("=" * tot_width)

    header_cols = "".join(
        f"{col:>{w}}" for col, w in zip(corr_matrix.columns, col_widths)
    )
    print(f"{'Method':<{name_width}}{'Group':<{group_width}}{header_cols}")
    print("-" * tot_width)

    prev_grp = None
    for (canon_key, display_name, grp), (_, r_vals), (_, r_p) in zip(
        models_meta, corr_matrix.iterrows(), p_matrix.iterrows()
    ):
        if prev_grp is not None and grp != prev_grp:
            print("-" * tot_width)
        prev_grp = grp

        cols_str = ""
        for col, w in zip(corr_matrix.columns, col_widths):
            val_str = format_sig_figs(r_vals[col], sig_digits)
            ast = get_asterisks(r_p[col])
            formatted = f"{val_str} {ast}".strip()
            cols_str += f"{formatted:>{w}}"

        print(f"{display_name:<{name_width}}{grp:<{group_width}}{cols_str}")

    print("=" * tot_width)
    print("Significance: * p < 0.05, ** p < 0.01, *** p < 0.001\n")


def print_condition_correlations(
    corr_matrix: pd.DataFrame,
    metric: str = "spearman",
    sig_digits: int = 3,
) -> None:
    """Print the pairwise condition-by-condition correlation matrix."""
    cond_corr = corr_matrix.corr()
    symbol = r"\rho" if metric == "spearman" else "r"
    print(
        f"--- Inter-Condition Correlation Matrix (corr of {metric} {symbol} across models) ---"
    )
    print(cond_corr.round(sig_digits).to_string())
    print()


def plot_correlation_matrix_heatmap(
    corr_matrix: pd.DataFrame,
    p_matrix: pd.DataFrame,
    models_meta: list[tuple[str, str, str]],
    metric: str = "spearman",
    output_path: Optional[Path] = None,
    cmap: str = "Blues",
    sig_digits: int = 3,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """Generate a publication-ready heatmap of the correlation matrix."""
    plt.rcParams.update(
        {
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
        }
    )

    n_rows, n_cols = corr_matrix.shape
    fig_w = 4.8 + n_cols * 1.1
    fig_h = 2.2 + n_rows * 0.48

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    data = corr_matrix.values
    im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # X-axis
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(corr_matrix.columns, fontsize=11, fontweight="bold")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Y-axis
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(corr_matrix.index, fontsize=10.5)

    # Annotate values and significance asterisks
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if pd.isna(val):
                continue
            val_str = format_sig_figs(val, sig_digits)
            p_val = p_matrix.iloc[i, j]
            ast = get_asterisks(p_val)
            text_val = f"{val_str}{ast}"

            # High-contrast text color based on cell brightness
            text_color = "white" if val > 0.65 else "#111111"
            ax.text(
                j,
                i,
                text_val,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9.5,
                fontweight="bold" if ast else "normal",
            )

    # Draw separator lines between method groups
    groups = [m[2] for m in models_meta]
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color="#555555", linewidth=1.2, linestyle="--")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    metric_symbol = r"$\rho$" if metric == "spearman" else "$r$"
    cbar.set_label(
        f"Group {metric.capitalize()} ({metric_symbol})",
        fontsize=10.5,
        fontweight="bold",
    )
    cbar.ax.tick_params(labelsize=9.5)

    metric_name = "Spearman" if metric == "spearman" else "Pearson"
    ax.set_title(
        f"Audio Distance Group Perceptual Correlation ({metric_name})",
        fontsize=12,
        fontweight="bold",
        pad=16,
    )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        print(f"Heatmap figure successfully saved to: {output_path}")

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
        description="Print correlation matrix and generate publication-ready heatmap for audio distance group correlations."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/correlation_results.tsv",
        help="Path to correlation_results.tsv (default: data/correlation_results.tsv)",
    )
    parser.add_argument(
        "-nc",
        "--noise-ceiling",
        default="data/noise_ceiling_results.tsv",
        help="Path to noise_ceiling_results.tsv (default: data/noise_ceiling_results.tsv)",
    )
    parser.add_argument(
        "-m",
        "--metric",
        choices=["spearman", "pearson", "both"],
        default="spearman",
        help="Correlation metric to display: 'spearman' (default), 'pearson', or 'both'.",
    )
    parser.add_argument(
        "--spearman",
        action="store_const",
        dest="metric",
        const="spearman",
        help="Shortcut to select Spearman group correlation.",
    )
    parser.add_argument(
        "--pearson",
        action="store_const",
        dest="metric",
        const="pearson",
        help="Shortcut to select Pearson group correlation.",
    )
    parser.add_argument(
        "--both",
        action="store_const",
        dest="metric",
        const="both",
        help="Shortcut to display both Spearman and Pearson correlation matrices.",
    )
    parser.add_argument(
        "--include-noise-ceiling",
        action="store_true",
        help="Include human noise ceiling benchmark as a reference row.",
    )
    parser.add_argument(
        "--inter-condition",
        "--pairwise",
        action="store_true",
        help="Also print inter-condition pairwise correlation matrix.",
    )
    parser.add_argument(
        "--sig-digits",
        "--sig-figs",
        type=int,
        default=3,
        dest="sig_digits",
        help="Number of significant digits to print (default: 3).",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["table", "markdown", "csv"],
        default="table",
        help="Output format for printing to terminal (default: table).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="../../out/figure_correlation_matrix.png",
        help="Path to save figure image (default: ../../out/figure_correlation_matrix.png).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate or save any plot (terminal output only).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plot interactively in a window (only save to file).",
    )
    parser.add_argument(
        "--cmap",
        default="Blues",
        help="Matplotlib colormap name for heatmap (default: Blues).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in dots per inch for raster export (default: 300).",
    )

    args = parser.parse_args()

    input_path = resolve_file_path(args.input)
    if not input_path.exists():
        sys.stderr.write(
            f"Error: Correlation results file not found at: {input_path}\n"
        )
        sys.exit(1)

    nc_path = resolve_file_path(args.noise_ceiling)

    metrics_to_run = ["spearman", "pearson"] if args.metric == "both" else [args.metric]

    for met in metrics_to_run:
        corr_matrix, p_matrix, models_meta = load_correlation_matrix(
            corr_path=input_path,
            nc_path=nc_path,
            metric=met,
            include_noise_ceiling=args.include_noise_ceiling,
        )

        print_correlation_matrix(
            corr_matrix=corr_matrix,
            p_matrix=p_matrix,
            models_meta=models_meta,
            metric=met,
            sig_digits=args.sig_digits,
            output_format=args.format,
        )

        if args.inter_condition:
            print_condition_correlations(
                corr_matrix=corr_matrix,
                metric=met,
                sig_digits=args.sig_digits,
            )

        if not args.no_plot:
            out_p = None
            if args.output:
                base_out = Path(args.output).expanduser().resolve()
                if len(metrics_to_run) > 1:
                    out_p = base_out.with_name(
                        f"{base_out.stem}_{met}{base_out.suffix}"
                    )
                else:
                    out_p = base_out

            plot_correlation_matrix_heatmap(
                corr_matrix=corr_matrix,
                p_matrix=p_matrix,
                models_meta=models_meta,
                metric=met,
                output_path=out_p,
                cmap=args.cmap,
                sig_digits=args.sig_digits,
                dpi=args.dpi,
                show=not args.no_show,
            )


if __name__ == "__main__":
    main()
