"""Interaction-Pooled ANOVA and Variance Decomposition for Audio Loss Functions.

Analyzes single-replicate distance measurements from data/distances.tsv.
Instead of treating randomized LFO phases as pseudo-subjects in a repeated-measures design,
this script treats each loss evaluation as an unreplicated factorial design across:
- modulation (3 levels: amp, freq, reg)
- feature (3 levels: brightness, richness, warmth)
- source (2 levels: real, synthetic)
- rating_stimulus / amount (4 non-reference stimulus levels: condition_a .. condition_d)

Total cells = 3 x 3 x 2 x 4 = 72 observations per loss function.

ANOVA is computed via statsmodels.api.stats.anova_lm.
Vectorized effect sizes (np2, eta_sq) match pingouin.anova implementations.
Multiple hypothesis correction (FDR via Benjamini-Hochberg) is computed via
statsmodels.stats.multitest.multipletests.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def parse_wavetable(wt: str) -> tuple[str, str]:
    """Extract feature and source from wavetable identifier."""
    parts = wt.split("__")[0].split("_")
    feature = parts[0]
    source = parts[1]
    return feature, source


def map_amount_to_condition(mod_type: str, amount: float) -> str:
    """Map numeric modulation amount to ordinal condition level (a, b, c, d)."""
    order_dict = {
        "amp": [0.3, 0.5, 0.7, 0.9],
        "freq": [0.5, 1.0, 2.0, 4.0],
        "reg": [0.125, 0.25, 0.375, 0.5],
    }
    amounts = order_dict.get(mod_type)
    if amounts is None:
        raise ValueError(f"Unknown modulation type: {mod_type}")
    if amount not in amounts:
        raise ValueError(f"Amount {amount} not valid for modulation type {mod_type}")
    else:
        idx = amounts.index(amount)
    conds = ["condition_a", "condition_b", "condition_c", "condition_d"]
    return conds[idx]


def prepare_loss_data(df: pd.DataFrame, loss_fn: str) -> pd.DataFrame:
    """Filter and format the 72 non-reference rows for a specific loss function."""
    sub = df[(df["loss_fn"] == loss_fn) & (~df["is_reference"])].copy()
    if sub.empty:
        raise ValueError(f"No non-reference data found for loss function: {loss_fn}")

    features, sources = zip(*[parse_wavetable(w) for w in sub["wavetable"]])
    sub["feature"] = features
    sub["source"] = sources
    sub["modulation"] = sub["mod_type"]
    sub["rating_stimulus"] = [
        map_amount_to_condition(m, a) for m, a in zip(sub["modulation"], sub["amount"])
    ]

    expected_rows = 3 * 3 * 2 * 4  # 72
    factors = ["modulation", "feature", "source", "rating_stimulus"]
    cell_counts = (
        sub.groupby(factors)
        .size()
    )
    if len(cell_counts) != expected_rows or not (cell_counts == 1).all():
        raise ValueError(
            f"{loss_fn}: expected exactly one observation per factorial cell; "
            f"found {len(cell_counts)} unique cells."
        )

    return sub


def compute_interaction_pooled_anova(
    df_loss: pd.DataFrame,
    max_interaction: int = 2,
    dv: str = "distance",
) -> pd.DataFrame:
    """Compute interaction-pooled ANOVA and variance decomposition using statsmodels.

    ANOVA table is computed via statsmodels.api.stats.anova_lm.
    Effect sizes (np2, eta_sq) and FDR p-value corrections are calculated using
    vectorized operations equivalent to pingouin.parametric.anovan and statsmodels.stats.multitest.
    """
    if max_interaction not in (2, 3):
        raise ValueError("max_interaction must be 2 or 3")

    formula = (
        f"{dv} ~ (C(modulation) + C(feature) + C(source) + C(rating_stimulus))**{max_interaction}"
    )
    model = ols(formula, data=df_loss).fit()

    # Core ANOVA table from statsmodels
    aov = sm.stats.anova_lm(model, typ=2).reset_index()
    aov = aov.rename(
        columns={
            "index": "Source",
            "sum_sq": "SS",
            "df": "DF",
            "PR(>F)": "p_unc",
        }
    )

    # Standardize factor names
    aov["Source"] = (
        aov["Source"]
        .str.replace("C(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("Residual", "Residual (pooled)", regex=False)
    )

    # Mean Squares: MS = SS / DF
    aov["MS"] = aov["SS"] / aov["DF"]

    # Degrees of freedom for residual
    resid_df = aov.loc[aov["Source"] == "Residual (pooled)", "DF"].values[0]

    # Partial eta-squared: np2 = (F * DF1) / (F * DF1 + DF2)  (Pingouin anovan standard)
    aov["np2"] = (aov["F"] * aov["DF"]) / (aov["F"] * aov["DF"] + resid_df)

    # Exact Variance Decomposition: eta_sq = SS / SS_total
    ss_total = aov["SS"].sum()
    aov["eta_sq"] = aov["SS"] / ss_total
    aov["pct_var"] = aov["eta_sq"] * 100.0

    # Multiple hypothesis testing corrections:
    # 1. Bonferroni (controls Family-Wise Error Rate, FWER)
    # 2. Benjamini-Hochberg (controls False Discovery Rate, FDR)
    # mask = aov["p_unc"].notna()
    mask = (
        aov["p_unc"].notna()
        & (aov["Source"] != "Residual (pooled)")
    )
    aov["p_bonf"] = np.nan
    aov["p_fdr"] = np.nan
    if mask.any():
        p_vals = aov.loc[mask, "p_unc"]
        aov.loc[mask, "p_bonf"] = multipletests(p_vals, method="bonferroni")[1]
        aov.loc[mask, "p_fdr"] = multipletests(p_vals, method="fdr_bh")[1]

    col_order = [
        "Source",
        "SS",
        "DF",
        "MS",
        "F",
        "p_unc",
        "p_bonf",
        "p_fdr",
        "np2",
        "eta_sq",
        "pct_var",
    ]
    return aov[col_order]


def format_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format ANOVA table to 4 decimal places with no scientific notation."""
    df_disp = df.copy()
    for col in ["SS", "MS", "F", "p_unc", "p_bonf", "p_fdr", "np2", "eta_sq", "pct_var"]:
        if col in df_disp.columns:
            df_disp[col] = df_disp[col].apply(
                lambda x: "" if pd.isna(x) else f"{x:.4f}"
            )
    df_disp["DF"] = df_disp["DF"].astype(int)
    return df_disp


def compute_human_variance_profile(
    data_path: Path,
    max_interaction: int = 2,
) -> dict[str, float]:
    """Compute the variance decomposition profile from the human listening study for comparison."""
    if not data_path.exists():
        return {}

    df_human = pd.read_csv(data_path, sep="\t")
    df_human = df_human[df_human["rating_stimulus"] != "reference"].copy()
    split_cols = df_human["trial_id"].str.split("_", expand=True)
    df_human["modulation"] = split_cols[0]
    df_human["feature"] = split_cols[1]
    df_human["source"] = split_cols[2]

    mod_means = (
        df_human.groupby(
            ["modulation", "feature", "source", "rating_stimulus"], as_index=False
        )["rating_score"]
        .mean()
        .rename(columns={"rating_score": "mean_rating"})
    )

    model = ols(
        f"mean_rating ~ (C(modulation) + C(feature) + C(source) + C(rating_stimulus))**{max_interaction}",
        data=mod_means,
    ).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    ss_total = ((mod_means["mean_rating"] - mod_means["mean_rating"].mean()) ** 2).sum()

    aov["eta_sq"] = aov["sum_sq"] / ss_total
    terms = {
        str(idx).replace("C(", "").replace(")", "").replace(" ", ""): val
        for idx, val in aov["eta_sq"].items()
    }

    two_way_keys = [k for k in terms if k.count(":") == 1]
    two_way_sum = sum(terms[k] for k in two_way_keys)

    profile = {
        "rating_stimulus": terms.get("rating_stimulus", 0.0),
        "modulation": terms.get("modulation", 0.0),
        "feature": terms.get("feature", 0.0),
        "source": terms.get("source", 0.0),
        "interactions_2way": two_way_sum,
    }
    if max_interaction >= 3:
        three_way_keys = [k for k in terms if k.count(":") == 2]
        three_way_sum = sum(terms[k] for k in three_way_keys)
        profile["interactions_3way"] = three_way_sum
    profile["residual_pooled"] = terms.get("Residual", 0.0)

    return profile


def run_variance_analysis(
    data_path: Path,
    output_path: Path | None = None,
    loss_fns: list[str] | None = None,
    max_interaction: int = 2,
):
    """Run interaction-pooled ANOVA and variance decomposition for loss functions."""
    log.info(f"Loading distance data from: {data_path}")
    df = pd.read_csv(data_path, sep="\t")

    all_losses = sorted(df["loss_fn"].unique())
    if loss_fns:
        selected_losses = [l for l in loss_fns if l in all_losses]
        missing = set(loss_fns) - set(selected_losses)
        if missing:
            log.warning(f"Requested loss functions not found in data: {missing}")
    else:
        selected_losses = all_losses

    log.info(
        f"Analyzing {len(selected_losses)} loss functions with max_interaction={max_interaction}..."
    )

    results_by_loss = {}
    summary_records = []

    for loss in selected_losses:
        sub = prepare_loss_data(df, loss)
        res_df = compute_interaction_pooled_anova(
            sub, max_interaction=max_interaction, dv="distance"
        )
        results_by_loss[loss] = res_df

        term_map = dict(zip(res_df["Source"], res_df["pct_var"]))
        two_way_terms = [k for k in term_map if k.count(":") == 1]
        two_way_pct = sum(term_map[k] for k in two_way_terms)

        rec = {
            "loss_fn": loss,
            "% Var (Amount)": term_map.get("rating_stimulus", 0.0),
            "% Var (Modulation)": term_map.get("modulation", 0.0),
            "% Var (Feature)": term_map.get("feature", 0.0),
            "% Var (Source)": term_map.get("source", 0.0),
            "% Var (2-Way Inter.)": two_way_pct,
        }
        if max_interaction >= 3:
            three_way_terms = [k for k in term_map if k.count(":") == 2]
            three_way_pct = sum(term_map[k] for k in three_way_terms)
            rec["% Var (3-Way Inter.)"] = three_way_pct
        rec["% Var (Residual)"] = term_map.get("Residual (pooled)", 0.0)

        summary_records.append(rec)

    sep_bar = "=" * 125
    log.info("\n" + sep_bar)
    log.info(f"INTERACTION-POOLED ANOVA & VARIANCE DECOMPOSITION (Pooled Order: >{max_interaction}-Way)")
    log.info(sep_bar)

    for loss in selected_losses:
        log.info(f"\n--- Loss Function: {loss} ---")
        disp_df = format_table_for_display(results_by_loss[loss])
        log.info("\n" + disp_df.to_string(index=False))

    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df.sort_values("% Var (Amount)", ascending=False).reset_index(
        drop=True
    )

    log.info("\n" + sep_bar)
    log.info("SUMMARY: VARIANCE DECOMPOSITION PROFILES (% OF TOTAL VARIANCE EXPLAINED)")
    log.info(sep_bar)

    repo_root = Path(__file__).resolve().parent.parent
    human_path = repo_root / "data" / "listening_test_responses_postprocessed.tsv"
    human_prof = compute_human_variance_profile(human_path, max_interaction=max_interaction)
    if human_prof:
        log.info("\n[Human Listening Study Reference Profile (% of mean rating variance)]:")
        h_dict = {
            "Reference": "Human Listeners",
            "% Var (Amount)": human_prof["rating_stimulus"] * 100,
            "% Var (Modulation)": human_prof["modulation"] * 100,
            "% Var (Feature)": human_prof["feature"] * 100,
            "% Var (Source)": human_prof["source"] * 100,
            "% Var (2-Way Inter.)": human_prof["interactions_2way"] * 100,
        }
        if max_interaction >= 3:
            h_dict["% Var (3-Way Inter.)"] = human_prof.get("interactions_3way", 0.0) * 100
        h_dict["% Var (Residual)"] = human_prof["residual_pooled"] * 100
        h_row = pd.DataFrame([h_dict])
        for c in h_row.columns[1:]:
            h_row[c] = h_row[c].apply(lambda x: f"{x:.2f}%")
        log.info(h_row.to_string(index=False))

    log.info("\n[Loss Function Profiles]:")
    disp_summary = summary_df.copy()
    for col in disp_summary.columns[1:]:
        disp_summary[col] = disp_summary[col].apply(lambda x: f"{x:.2f}%")
    log.info(disp_summary.to_string(index=False))
    log.info(sep_bar + "\n")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_records = []
        for loss, df_res in results_by_loss.items():
            df_copy = df_res.copy()
            df_copy.insert(0, "loss_fn", loss)
            combined_records.append(df_copy)
        combined_df = pd.concat(combined_records, ignore_index=True)
        combined_df.to_csv(output_path, sep="\t", index=False)
        log.info(f"Successfully saved detailed ANOVA & variance results to: {output_path}")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_data_path = repo_root / "data" / "distances.tsv"
    default_out_path = repo_root / "data" / "distances_loss_variance_analysis.tsv"

    parser = argparse.ArgumentParser(
        description="Interaction-pooled ANOVA and variance decomposition for audio loss functions."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=str(default_data_path),
        help=f"Path to distances TSV file (default: {default_data_path})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(default_out_path),
        help=f"Path to save output results TSV (default: {default_out_path})",
    )
    parser.add_argument(
        "--loss-fn",
        "--loss-fns",
        "-l",
        nargs="+",
        # default=None,
        default=["jtfs_log1p", "scat1d_log1p", "vggish", "panns_wavegram_logmel", "clap2", "encodec48k", "mss_rev", "mss_log_lin", "mfcc"],
        help="Filter analysis to specific loss function(s) (e.g. -l mfcc mss_log_lin)",
    )
    parser.add_argument(
        "--max-interaction",
        type=int,
        default=2,
        # default=3,
        choices=[2, 3],
        help="Maximum interaction order to include (2: pool 3-way & 4-way, 3: pool 4-way; default: 2)",
    )
    args = parser.parse_args()

    run_variance_analysis(
        data_path=Path(args.data_path),
        output_path=Path(args.output) if args.output else None,
        loss_fns=args.loss_fn,
        max_interaction=args.max_interaction,
    )


if __name__ == "__main__":
    main()
