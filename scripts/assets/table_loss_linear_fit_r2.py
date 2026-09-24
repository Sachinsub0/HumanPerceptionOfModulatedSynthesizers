"""Generate publication-ready LaTeX table for loss function linear fit determination coefficients (R^2).

Reads distances dataset (and optionally human listening test responses)
and outputs a formatted LaTeX table comparing the R^2 values across
amplitude, frequency, and irregularity modulations against human listeners
as the perceptual benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from scripts.assets.figure_loss_linear_fit import (
        generate_r2_latex_table,
        load_distances_data,
        load_human_data,
        normalize_loss_fns,
        resolve_file_path,
        resolve_output_path,
    )
except ImportError:
    from figure_loss_linear_fit import (  # type: ignore
        generate_r2_latex_table,
        load_distances_data,
        load_human_data,
        normalize_loss_fns,
        resolve_file_path,
        resolve_output_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Print compact LaTeX table of loss function linear fit R^2 determination coefficients."
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
        dest="human_data",
        help="Optional path to postprocessed human listening responses TSV (default: data/listening_test_responses_postprocessed.tsv).",
    )
    parser.add_argument(
        "--loss-fn",
        "--loss-fns",
        "-l",
        nargs="+",
        default=None,
        help="Filter table to specific loss function(s) (e.g. -l mfcc mss_log_lin).",
    )
    parser.add_argument(
        "--anchor-zero",
        "--anchor",
        action="store_true",
        dest="anchor_zero",
        default=True,
        help="Anchor linear fit to pass through y=0 at first x value (default: True).",
    )
    parser.add_argument(
        "--no-anchor-zero",
        "--no-anchor",
        action="store_false",
        dest="anchor_zero",
        help="Compute standard unconstrained linear fit R^2 = (Pearson's r)^2.",
    )
    parser.add_argument(
        "--sig-digits",
        "--precision",
        type=int,
        default=3,
        dest="sig_digits",
        help="Number of significant digits / decimal places (default: 3).",
    )
    parser.add_argument(
        "--no-mean",
        action="store_true",
        dest="no_mean",
        default=False,
        help="Omit the summary Mean column.",
    )
    parser.add_argument(
        "--table-env",
        choices=["table*", "table"],
        default="table*",
        help="LaTeX table environment wrapper: 'table*' (default) or 'table'.",
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
        sys.stderr.write(f"Error: Distances data file not found at: {input_path}\n")
        sys.exit(1)

    df_dist = load_distances_data(input_path)
    df_human = None
    if args.human_data:
        human_path = resolve_file_path(args.human_data)
        if human_path.exists():
            df_human = load_human_data(human_path)

    loss_fns_clean = normalize_loss_fns(args.loss_fn)

    tex_table = generate_r2_latex_table(
        df=df_dist,
        df_human=df_human,
        anchor_zero=args.anchor_zero,
        sig_digits=args.sig_digits,
        include_mean=not args.no_mean,
        table_env=args.table_env,
        loss_fns=loss_fns_clean,
    )

    if not args.quiet:
        print(tex_table)

    if args.output:
        out_path = resolve_output_path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(tex_table, encoding="utf-8")
        sys.stderr.write(f"Table successfully written to: {out_path}\n")


if __name__ == "__main__":
    main()
