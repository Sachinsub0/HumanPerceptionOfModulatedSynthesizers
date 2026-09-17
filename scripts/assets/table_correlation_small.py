"""Generate publication-ready compact LaTeX table for audio distance group correlations.

Reads data/correlation_results.tsv and data/noise_ceiling_results.tsv
and prints a formatted LaTeX table comparing audio distance models across
amplitude, frequency, and irregularity modulations against human listeners
and the human noise ceiling benchmark (group correlations only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CONDITIONS = ["amp", "freq", "reg"]

CONDITION_MAP = {
    "amp": "Amplitude",
    "freq": "Frequency",
    "reg": "Irregularity",
}

METHOD_GROUPS = [
    [
        ("mss_log_lin", "MSS Log + Linear"),
        ("mss_rev", "MSS Revisited"),
        ("mfcc", "MFCC"),
    ],
    [
        ("scat1d_log1p", "Scat1D"),
        ("jtfs_log1p", "JTFS"),
    ],
    [
        ("vggish", "VGGish"),
        ("encodec48k", "EnCodec 48kHz"),
        ("clap2", "MS-CLAP"),
        ("panns_wavegram_logmel", "PANNs WGLM"),
    ],
]

METHOD_GROUP_NAMES = [
    "STFT--based",
    "Wavelet--based",
    "Neural--based",
]

METHOD_GROUP_TAGS = [
    r"$^{\;\mathcal{S}}$",
    r"$^{\;\mathcal{W}}$",
    r"$^{\;\mathcal{N}}$",
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


def format_sig_figs(val: float, precision: int) -> str:
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
    """Return LaTeX significance asterisks based on p-value."""
    if p_val is None or pd.isna(p_val):
        return ""
    if p_val < 0.001:
        return "$^{***}$"
    elif p_val < 0.01:
        return "$^{**}$"
    elif p_val < 0.05:
        return "$^{*}$"
    return ""


def assign_highlights(
    model_entries: list[tuple[str, float, str]],
) -> dict[str, str]:
    """Identify top 2 models per metric.

    Returns dict mapping model_key -> formatted string with \\textbf or \\underline.
    Handles ties in rounded string or raw float values.
    """
    if not model_entries:
        return {}

    sorted_by_val = sorted(model_entries, key=lambda x: x[1], reverse=True)
    top1_key, top1_val, top1_str = sorted_by_val[0]

    top2_key, top2_val, top2_str = None, None, None
    if len(sorted_by_val) > 1:
        top2_key, top2_val, top2_str = sorted_by_val[1]

    res = {}
    for key, val, s_val in model_entries:
        if (
            key == top1_key
            or val == top1_val
            or (s_val == top1_str and s_val != top2_str)
        ):
            res[key] = f"{{\\textbf{{{s_val}}}}}"
        elif (
            key == top2_key
            or val == top2_val
            or (s_val == top2_str and s_val != top1_str)
        ):
            res[key] = f"{{\\underline{{{s_val}}}}}"
        elif s_val == top1_str and top2_str == top1_str:
            if key == sorted_by_val[0][0]:
                res[key] = f"{{\\textbf{{{s_val}}}}}"
            elif key == sorted_by_val[1][0] or (
                len(sorted_by_val) > 2 and key == sorted_by_val[2][0]
            ):
                res[key] = f"{{\\underline{{{s_val}}}}}"
            else:
                res[key] = s_val
        else:
            res[key] = s_val

    return res


def generate_latex_table(
    df: pd.DataFrame,
    nc_df: pd.DataFrame,
    sig_digits: int = 3,
    group_means: bool = False,
) -> str:
    """Generate the compact LaTeX table string from correlation and noise ceiling dataframes."""
    df = df.copy()
    if "loss_fn" in df.columns:
        df["canonical_loss"] = df["loss_fn"].map(lambda x: LOSS_FN_ALIASES.get(x, x))

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{")
    lines.append(
        r"Audio distance correlation with human group perceptual dissimilarity across amplitude, "
        r"frequency, and irregularity modulations. The human noise ceiling acts as the reference upper bound, "
        r"reported as its 95\% bootstrap confidence interval $[95\%\text{ CI}]$. "
        r"For loss functions, the highest correlation within each modulation condition is in \textbf{bold}, "
        r"and the second highest is \underline{underlined}. "
        r"Statistical significance is denoted by asterisks ($^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$; "
        r"unstarred indicates $p \ge 0.05$)."
    )
    lines.append(r"}")
    lines.append(r"\label{tab:audio_distance_group_perceptual_correlation}")
    lines.append(r"\vspace{3pt}")
    lines.append(r"\sisetup{")
    lines.append(r"    round-pad = true,")
    lines.append(r"    reset-text-series=false, ")
    lines.append(r"    text-series-to-math=true, ")
    lines.append(r"    mode=text,")
    lines.append(r"    tight-spacing=false,")
    lines.append(r"    separate-uncertainty=false,")
    lines.append(r"    detect-weight=true,")
    lines.append(r"    detect-inline-weight=math,")
    lines.append(r"}")
    lines.append(r"\setlength{\tabcolsep}{4.5pt}")
    lines.append(r"\begin{tabular}{")
    lines.append(r"    l")
    for _ in range(6):
        lines.append(
            f"    S[round-mode=figures, round-precision={sig_digits}, table-format=1.{sig_digits}, table-number-alignment=right] @{{\\,}} l"
        )
    lines.append(r"}")
    lines.append(r"    \toprule")
    lines.append(r"    \multirow[t]{2}{*}{\textbf{Method}} ")
    lines.append(r"        & \multicolumn{4}{c}{\textbf{Amplitude}} ")
    lines.append(r"        & \multicolumn{4}{c}{\textbf{Frequency}} ")
    lines.append(r"        & \multicolumn{4}{c}{\textbf{Irregularity}} \\")
    lines.append(r"    \cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}")
    lines.append(
        r"        & \multicolumn{2}{c}{Spearman ($\rho$) $\uparrow$} & \multicolumn{2}{c}{Pearson ($r$) $\uparrow$} "
    )
    lines.append(
        r"        & \multicolumn{2}{c}{Spearman ($\rho$) $\uparrow$} & \multicolumn{2}{c}{Pearson ($r$) $\uparrow$} "
    )
    lines.append(
        r"        & \multicolumn{2}{c}{Spearman ($\rho$) $\uparrow$} & \multicolumn{2}{c}{Pearson ($r$) $\uparrow$} \\"
    )
    lines.append(r"    \midrule")

    # Noise ceiling row across all 3 conditions
    lines.append("    Noise Ceiling ")
    for cond_idx, cond in enumerate(CONDITIONS):
        nc_match = nc_df[
            (nc_df["granularity"] == "modulation") & (nc_df["condition"] == cond)
        ]
        if nc_match.empty:
            raise ValueError(
                f"No noise ceiling row found for condition '{cond}' in noise ceiling data."
            )
        nc_data = nc_match.iloc[0]

        sp_ci_low = format_sig_figs(nc_data["spearman_group_ci95_low"], sig_digits)
        sp_ci_high = format_sig_figs(nc_data["spearman_group_ci95_high"], sig_digits)
        sp_ci = f"$[{sp_ci_low}, {sp_ci_high}]$"

        pe_ci_low = format_sig_figs(nc_data["pearson_group_ci95_low"], sig_digits)
        pe_ci_high = format_sig_figs(nc_data["pearson_group_ci95_high"], sig_digits)
        pe_ci = f"$[{pe_ci_low}, {pe_ci_high}]$"

        line_term = "\\\\" if cond_idx == len(CONDITIONS) - 1 else ""
        lines.append(
            f"        & \\multicolumn{{2}}{{c}}{{{sp_ci}}} & \\multicolumn{{2}}{{c}}{{{pe_ci}}} {line_term}"
        )

    lines.append(r"    \midrule")

    # Compute highlights for each condition independently
    hl_by_cond = {}
    cond_dfs = {}
    for cond in CONDITIONS:
        c_df = df[(df["condition"] == cond) & (df["granularity"] == "modulation")]
        cond_dfs[cond] = c_df

        sp_entries = []
        pe_entries = []
        for group in METHOD_GROUPS:
            for canon_key, _ in group:
                match = c_df[c_df["canonical_loss"] == canon_key]
                if match.empty:
                    match = c_df[c_df["loss_fn"] == canon_key]
                if not match.empty:
                    r = match.iloc[0]
                    sp_entries.append(
                        (
                            canon_key,
                            float(r["spearman_group"]),
                            format_sig_figs(r["spearman_group"], sig_digits),
                        )
                    )
                    pe_entries.append(
                        (
                            canon_key,
                            float(r["pearson_group"]),
                            format_sig_figs(r["pearson_group"], sig_digits),
                        )
                    )

        hl_by_cond[cond] = {
            "sp": assign_highlights(sp_entries),
            "pe": assign_highlights(pe_entries),
        }

    # Render each model across the 3 conditions
    for g_idx, group in enumerate(METHOD_GROUPS):
        tag = METHOD_GROUP_TAGS[g_idx]
        if g_idx > 0:
            lines.append(r"    \addlinespace[5pt]")
        for canon_key, display_name in group:
            full_display_name = f"{display_name}{tag}"
            lines.append(f"    {full_display_name:<32s}")
            for cond_idx, cond in enumerate(CONDITIONS):
                c_df = cond_dfs[cond]
                match = c_df[c_df["canonical_loss"] == canon_key]
                if match.empty:
                    match = c_df[c_df["loss_fn"] == canon_key]
                if match.empty:
                    continue
                row = match.iloc[0]

                sp_val = hl_by_cond[cond]["sp"].get(
                    canon_key, format_sig_figs(row["spearman_group"], sig_digits)
                )
                sp_ast = get_asterisks(row["spearman_group_p"])
                pe_val = hl_by_cond[cond]["pe"].get(
                    canon_key, format_sig_figs(row["pearson_group"], sig_digits)
                )
                pe_ast = get_asterisks(row["pearson_group_p"])

                sp_ast_pad = f"{sp_ast:<15s}" if sp_ast else " " * 15
                pe_ast_pad = f"{pe_ast:<15s}" if pe_ast else " " * 15

                line_term = "\\\\" if cond_idx == len(CONDITIONS) - 1 else ""
                lines.append(
                    f"        & {sp_val} & {sp_ast_pad}& {pe_val} & {pe_ast_pad}{line_term}"
                )

    # Optional 3 extra rows with group means
    if group_means:
        lines.append(r"    \midrule")
        # Compute highlights among the 3 method group means for each condition independently
        group_means_hl = {}
        for cond in CONDITIONS:
            c_df = cond_dfs[cond]
            sp_entries = []
            pe_entries = []
            for g_idx, group in enumerate(METHOD_GROUPS):
                g_name = METHOD_GROUP_NAMES[g_idx]
                sp_vals = []
                pe_vals = []
                for canon_key, _ in group:
                    match = c_df[c_df["canonical_loss"] == canon_key]
                    if match.empty:
                        match = c_df[c_df["loss_fn"] == canon_key]
                    if not match.empty:
                        sp_vals.append(float(match.iloc[0]["spearman_group"]))
                        pe_vals.append(float(match.iloc[0]["pearson_group"]))

                sp_mean = sum(sp_vals) / len(sp_vals) if sp_vals else float("nan")
                pe_mean = sum(pe_vals) / len(pe_vals) if pe_vals else float("nan")

                sp_entries.append(
                    (g_name, sp_mean, format_sig_figs(sp_mean, sig_digits))
                )
                pe_entries.append(
                    (g_name, pe_mean, format_sig_figs(pe_mean, sig_digits))
                )

            group_means_hl[cond] = {
                "sp": assign_highlights(sp_entries),
                "pe": assign_highlights(pe_entries),
            }

        for g_idx, group_name in enumerate(METHOD_GROUP_NAMES):
            tag = METHOD_GROUP_TAGS[g_idx]
            full_group_name = f"{group_name}{tag}"
            lines.append(f"    {full_group_name:<32s}")
            for cond_idx, cond in enumerate(CONDITIONS):
                sp_str = group_means_hl[cond]["sp"].get(group_name, "")
                pe_str = group_means_hl[cond]["pe"].get(group_name, "")

                line_term = "\\\\" if cond_idx == len(CONDITIONS) - 1 else ""
                lines.append(
                    f"        & \\multicolumn{{2}}{{l}}{{{sp_str}}} & \\multicolumn{{2}}{{l}}{{{pe_str}}} {line_term}"
                )

    lines.append(r"    \bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


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
        description="Print compact LaTeX group correlation table to terminal."
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
        "--sig-digits",
        "--sig-figs",
        type=int,
        default=3,
        dest="sig_digits",
        help="Number of significant digits for correlation coefficients and CIs (default: 3).",
    )
    parser.add_argument(
        "--group-means",
        "--means",
        action="store_true",
        default=True,
        dest="group_means",
        help="Display 3 extra rows with the mean for each method group (STFT-based, Wavelet-based, and Neural-based) at the bottom of the table after a midbar.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional path to save generated LaTeX table to a .tex file.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress printing to terminal (useful when exporting only to file).",
    )
    args = parser.parse_args()

    input_path = resolve_file_path(args.input)
    if not input_path.exists():
        sys.stderr.write(
            f"Error: Correlation results file not found at: {input_path}\n"
        )
        sys.exit(1)

    nc_path = resolve_file_path(args.noise_ceiling)
    if not nc_path.exists():
        sys.stderr.write(f"Error: Noise ceiling results file not found at: {nc_path}\n")
        sys.exit(1)

    sep_in = "\t" if input_path.suffix in [".tsv", ".txt"] else ","
    df = pd.read_csv(input_path, sep=sep_in)

    sep_nc = "\t" if nc_path.suffix in [".tsv", ".txt"] else ","
    nc_df = pd.read_csv(nc_path, sep=sep_nc)

    table_latex = generate_latex_table(
        df=df,
        nc_df=nc_df,
        sig_digits=args.sig_digits,
        group_means=args.group_means,
    )

    if not args.quiet:
        print(table_latex)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table_latex, encoding="utf-8")
        sys.stderr.write(f"Table successfully written to: {out_path}\n")


if __name__ == "__main__":
    main()
