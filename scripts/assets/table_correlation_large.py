"""Generate publication-ready LaTeX table for audio distance correlations.

Reads data/correlation_results.tsv and data/noise_ceiling_results.tsv
and prints a formatted LaTeX table comparing audio distance models across
amplitude, frequency, and irregularity modulations against human listeners
and the human noise ceiling benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Method mappings and ordering
CONDITION_MAP = {
    "amp": "Amplitude",
    "freq": "Frequency",
    "reg": "Irregularity",
}

METHOD_GROUPS = [
    [
        ("mss_log_lin", "MSS Log + Lin."),
        ("mss_rev", "MSS Revisited"),
        ("mfcc", "MFCC"),
    ],
    [
        ("scat1d_log1p", "Scat1D"),
        ("jtfs_log1p", "JTFS"),
    ],
    [
        ("vggish", "VGGish"),
        ("encodec48k", "EnCodec 48~kHz"),
        ("clap2", "MS-CLAP"),
        ("panns_wavegram_logmel", "PANNs WGLM"),
    ],
]

# Alternative loss function keys if different naming conventions are present in TSV
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

    # Sort descending by raw float value
    sorted_by_val = sorted(model_entries, key=lambda x: x[1], reverse=True)
    top1_key, top1_val, top1_str = sorted_by_val[0]

    top2_key, top2_val, top2_str = None, None, None
    if len(sorted_by_val) > 1:
        top2_key, top2_val, top2_str = sorted_by_val[1]

    res = {}
    for key, val, s_val in model_entries:
        # Check if top 1
        if key == top1_key or val == top1_val or (s_val == top1_str and s_val != top2_str):
            res[key] = f"{{\\textbf{{{s_val}}}}}"
        # Check if top 2
        elif key == top2_key or val == top2_val or (s_val == top2_str and s_val != top1_str):
            res[key] = f"{{\\underline{{{s_val}}}}}"
        # Special tie case: top 1 and top 2 share same rounded string
        elif s_val == top1_str and top2_str == top1_str:
            if key == sorted_by_val[0][0]:
                res[key] = f"{{\\textbf{{{s_val}}}}}"
            elif key == sorted_by_val[1][0] or (len(sorted_by_val) > 2 and key == sorted_by_val[2][0]):
                res[key] = f"{{\\underline{{{s_val}}}}}"
            else:
                res[key] = s_val
        else:
            res[key] = s_val

    return res


def generate_latex_table(
    df: pd.DataFrame,
    nc_df: pd.DataFrame,
    sig_digits_corr: int = 3,
    sig_digits_std: int = 2,
) -> str:
    """Generate the complete LaTeX table string from correlation and noise ceiling dataframes."""
    df = df.copy()
    if "loss_fn" in df.columns:
        df["canonical_loss"] = df["loss_fn"].map(lambda x: LOSS_FN_ALIASES.get(x, x))

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{")
    lines.append(
        r"Audio distance correlation with human perceptual dissimilarity across amplitude, "
        r"frequency, and irregularity modulations. The human noise ceiling acts as the reference upper bound, "
        r"reported as its 95\% bootstrap confidence interval $[95\%\text{ CI}]$. "
        r"For models, the highest correlation within each modulation condition is in \textbf{bold}, "
        r"and the second highest is \underline{underlined}. "
        r"Statistical significance is denoted by asterisks ($^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$; "
        r"unstarred indicates $p \ge 0.05$). Group columns report group correlation alongside significance asterisks "
        r"(or the 95\% CI for the noise ceiling); individual columns report mean correlation and standard deviation ($\mu \pm \sigma$)."
    )
    lines.append(r"}")
    lines.append(r"\label{tab:audio_distance_perceptual_correlation}")
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
    lines.append(r"\begin{tabular}{")
    lines.append(r"    l")
    lines.append(r"    l")
    lines.append(
        f"    S[round-mode=figures, round-precision={sig_digits_corr}, table-format=1.{sig_digits_corr}, table-number-alignment=right] @{{\\,}} l"
    )
    lines.append(
        f"    S[round-mode=figures, round-precision={sig_digits_corr}, table-format=1.{sig_digits_corr}, table-number-alignment=right] @{{\\hspace{{8pt}}$\\pm$\\hspace{{4pt}}}} "
    )
    lines.append(
        f"    S[round-mode=figures, round-precision={sig_digits_std}, table-format=1.{sig_digits_std}, table-number-alignment=left]"
    )
    lines.append(
        f"    S[round-mode=figures, round-precision={sig_digits_corr}, table-format=1.{sig_digits_corr}, table-number-alignment=right] @{{\\,}} l"
    )
    lines.append(
        f"    S[round-mode=figures, round-precision={sig_digits_corr}, table-format=1.{sig_digits_corr}, table-number-alignment=right] @{{\\hspace{{8pt}}$\\pm$\\hspace{{4pt}}}} "
    )
    lines.append(
        f"    S[round-mode=figures, round-precision={sig_digits_std}, table-format=1.{sig_digits_std}, table-number-alignment=left]"
    )
    lines.append(r"}")
    lines.append(r"    \toprule")
    lines.append(
        r"    \multirow[t]{2}{*}{\textbf{Mod. Type}} & \multirow[t]{2}{*}{\textbf{Method}} & "
        r"\multicolumn{4}{c}{\textbf{Spearman} ($\rho$) $\uparrow$} & "
        r"\multicolumn{4}{c}{\textbf{Pearson} ($r$) $\uparrow$} \\"
    )
    lines.append(r"    \cmidrule(lr){3-6} \cmidrule(lr){7-10}")
    lines.append(
        r"    & & \multicolumn{2}{c}{Group} & \multicolumn{2}{c}{Indiv. ($\mu \pm \sigma$)} "
    )
    lines.append(
        r"    & \multicolumn{2}{c}{Group} & \multicolumn{2}{c}{Indiv. ($\mu \pm \sigma$)} \\"
    )

    conditions = ["amp", "freq", "reg"]
    for cond in conditions:
        lines.append(r"    \midrule")
        cond_label = CONDITION_MAP[cond]
        lines.append(f"    \\multirow{{10}}{{*}}{{{cond_label}}}")

        # Noise ceiling row from noise_ceiling_results.tsv
        nc_match = nc_df[(nc_df["granularity"] == "modulation") & (nc_df["condition"] == cond)]
        if nc_match.empty:
            raise ValueError(f"No noise ceiling row found for condition '{cond}' in noise ceiling data.")
        nc_row = nc_match.iloc[0]

        # 95% bootstrap CI for group correlation
        sp_ci_low_str = format_sig_figs(nc_row["spearman_group_ci95_low"], sig_digits_corr)
        sp_ci_high_str = format_sig_figs(nc_row["spearman_group_ci95_high"], sig_digits_corr)
        sp_ci = f"$[{sp_ci_low_str}, {sp_ci_high_str}]$"

        pe_ci_low_str = format_sig_figs(nc_row["pearson_group_ci95_low"], sig_digits_corr)
        pe_ci_high_str = format_sig_figs(nc_row["pearson_group_ci95_high"], sig_digits_corr)
        pe_ci = f"$[{pe_ci_low_str}, {pe_ci_high_str}]$"

        # Lower noise ceiling for individual results (mean +/- std)
        sp_ind = format_sig_figs(nc_row["spearman_indiv_lower"], sig_digits_corr)
        sp_std = format_sig_figs(nc_row["spearman_indiv_lower_std"], sig_digits_std)
        pe_ind = format_sig_figs(nc_row["pearson_indiv_lower"], sig_digits_corr)
        pe_std = format_sig_figs(nc_row["pearson_indiv_lower_std"], sig_digits_std)

        lines.append(
            f"        & Noise Ceiling    & \\multicolumn{{2}}{{c}}{{{sp_ci}}} & {sp_ind:5s} & {sp_std:5s}"
            f"  & \\multicolumn{{2}}{{c}}{{{pe_ci}}} & {pe_ind:5s} & {pe_std:5s}  \\\\"
        )
        lines.append(r"        \cmidrule(lr){2-10}")

        # Extract model values for this condition
        cond_df = df[(df["condition"] == cond) & (df["granularity"] == "modulation")]
        cond_models_data = {}
        for group in METHOD_GROUPS:
            for canon_key, _ in group:
                match = cond_df[cond_df["canonical_loss"] == canon_key]
                if match.empty:
                    match = cond_df[cond_df["loss_fn"] == canon_key]
                if not match.empty:
                    cond_models_data[canon_key] = match.iloc[0]

        # Prepare lists for highlight calculations across all models in this condition
        sp_grp_entries = []
        sp_ind_entries = []
        pe_grp_entries = []
        pe_ind_entries = []

        for group in METHOD_GROUPS:
            for canon_key, _ in group:
                if canon_key in cond_models_data:
                    row = cond_models_data[canon_key]
                    sp_grp_entries.append(
                        (canon_key, float(row["spearman_group"]), format_sig_figs(row["spearman_group"], sig_digits_corr))
                    )
                    sp_ind_entries.append(
                        (canon_key, float(row["spearman_indiv"]), format_sig_figs(row["spearman_indiv"], sig_digits_corr))
                    )
                    pe_grp_entries.append(
                        (canon_key, float(row["pearson_group"]), format_sig_figs(row["pearson_group"], sig_digits_corr))
                    )
                    pe_ind_entries.append(
                        (canon_key, float(row["pearson_indiv"]), format_sig_figs(row["pearson_indiv"], sig_digits_corr))
                    )

        hl_sp_grp = assign_highlights(sp_grp_entries)
        hl_sp_ind = assign_highlights(sp_ind_entries)
        hl_pe_grp = assign_highlights(pe_grp_entries)
        hl_pe_ind = assign_highlights(pe_ind_entries)

        # Render each model group
        for g_idx, group in enumerate(METHOD_GROUPS):
            if g_idx > 0:
                lines.append(r"        \addlinespace[5pt]")
            for canon_key, display_name in group:
                if canon_key not in cond_models_data:
                    continue
                row = cond_models_data[canon_key]
                sp_grp_str = hl_sp_grp.get(canon_key, format_sig_figs(row["spearman_group"], sig_digits_corr))
                sp_p_str = get_asterisks(row["spearman_group_p"])
                sp_ind_str = hl_sp_ind.get(canon_key, format_sig_figs(row["spearman_indiv"], sig_digits_corr))
                sp_std_str = format_sig_figs(row["spearman_indiv_std"], sig_digits_std)

                pe_grp_str = hl_pe_grp.get(canon_key, format_sig_figs(row["pearson_group"], sig_digits_corr))
                pe_p_str = get_asterisks(row["pearson_group_p"])
                pe_ind_str = hl_pe_ind.get(canon_key, format_sig_figs(row["pearson_indiv"], sig_digits_corr))
                pe_std_str = format_sig_figs(row["pearson_indiv_std"], sig_digits_std)

                m_name_pad = f"{display_name:<18s}"
                sp_ast_pad = f"{sp_p_str:<17s}" if sp_p_str else " " * 18
                pe_ast_pad = f"{pe_p_str:<17s}" if pe_p_str else " " * 18

                lines.append(
                    f"        & {m_name_pad}& {sp_grp_str} & {sp_ast_pad}& {sp_ind_str} & {sp_std_str:<5s}"
                    f" & {pe_grp_str} & {pe_ast_pad}& {pe_ind_str} & {pe_std_str:<5s} \\\\"
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
        description="Print LaTeX correlation table from correlation_results.tsv and noise_ceiling_results.tsv to terminal."
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
        dest="sig_digits_corr",
        help="Number of significant digits for correlation coefficients and CIs (default: 3).",
    )
    parser.add_argument(
        "--sig-digits-std",
        "--sig-figs-std",
        type=int,
        default=2,
        dest="sig_digits_std",
        help="Number of significant digits for standard deviations (default: 2).",
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
        sys.stderr.write(f"Error: Correlation results file not found at: {input_path}\n")
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
        sig_digits_corr=args.sig_digits_corr,
        sig_digits_std=args.sig_digits_std,
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
