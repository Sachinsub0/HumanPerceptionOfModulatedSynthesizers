"""Interaction-Pooled ANOVA and Variance Decomposition for Audio Loss Functions.

Analyzes single-replicate distance measurements from data/distances.tsv alongside
human listening study mean ratings from data/listening_test_responses_postprocessed.tsv.

Instead of treating randomized LFO phases as pseudo-subjects in a repeated-measures design,
this script treats each metric evaluation as an unreplicated factorial design across:
- modulation (3 levels: amp, freq, reg)
- feature (3 levels: brightness, richness, warmth)
- source (2 levels: real, synthetic)
- rating_stimulus / amount (4 non-reference stimulus levels: condition_a .. condition_d)

Total cells = 3 x 3 x 2 x 4 = 72 observations per evaluation metric.

Optionally allows pooling ratings / distances across one or more factors before
computing and displaying the variance analysis (e.g. pooling across feature and source
leaves only modulation and rating_stimulus as main factors).

ANOVA is computed via statsmodels.api.stats.anova_lm.
Vectorized effect sizes (np2, eta_sq) match pingouin.anova implementations.
Multiple hypothesis corrections (Bonferroni & Benjamini-Hochberg FDR) are computed via
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

ALL_FACTORS: list[str] = ["modulation", "feature", "source", "rating_stimulus"]

FACTOR_ALIASES: dict[str, str] = {
    "modulation": "modulation",
    "mod": "modulation",
    "mod_type": "modulation",
    "feature": "feature",
    "feat": "feature",
    "source": "source",
    "src": "source",
    "rating_stimulus": "rating_stimulus",
    "stimulus": "rating_stimulus",
    "amount": "rating_stimulus",
    "amt": "rating_stimulus",
    "rating": "rating_stimulus",
}

FACTOR_DISPLAY_NAMES: dict[str, str] = {
    "rating_stimulus": "% Var (Amount)",
    "modulation": "% Var (Modulation)",
    "feature": "% Var (Feature)",
    "source": "% Var (Source)",
}

DISPLAY_FACTOR_ORDER: list[str] = [
    "rating_stimulus",
    "modulation",
    "feature",
    "source",
]


def normalize_pool_factors(raw_factors: list[str] | None) -> list[str]:
    """Normalize and validate factors to pool over."""
    if not raw_factors:
        return []

    pooled: list[str] = []
    for item in raw_factors:
        for f in item.replace(",", " ").split():
            clean = f.strip().lower()
            if not clean:
                continue
            if clean not in FACTOR_ALIASES:
                valid = sorted(set(FACTOR_ALIASES.values()))
                raise ValueError(
                    f"Unknown factor '{clean}'. Valid factors are: {valid} "
                    f"(aliases accepted: {list(FACTOR_ALIASES.keys())})"
                )
            canon = FACTOR_ALIASES[clean]
            if canon not in pooled:
                pooled.append(canon)

    remaining = [f for f in ALL_FACTORS if f not in pooled]
    if len(remaining) < 2:
        raise ValueError(
            f"Cannot pool over {pooled}. At least 2 factors must remain for unreplicated factorial ANOVA, "
            f"but only {len(remaining)} remained: {remaining}."
        )

    return pooled


def pool_data(df_data: pd.DataFrame, remaining_factors: list[str]) -> pd.DataFrame:
    """Pool ratings/distances across non-selected factors by taking cell means across remaining factors."""
    if set(remaining_factors) == set(ALL_FACTORS):
        return df_data

    pooled = (
        df_data.groupby(remaining_factors, as_index=False)["distance"]
        .mean()
    )
    if "loss_fn" in df_data.columns:
        pooled["loss_fn"] = df_data["loss_fn"].iloc[0]

    return pooled


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


def load_human_data(data_path: Path) -> pd.DataFrame:
    """Aggregate human listening test responses into the standard 72-cell factorial format."""
    df_human = pd.read_csv(data_path, sep="\t")
    df_human = df_human[df_human["rating_stimulus"] != "reference"].copy()

    split_cols = df_human["trial_id"].str.split("_", expand=True)
    df_human["modulation"] = split_cols[0]
    df_human["feature"] = split_cols[1]
    df_human["source"] = split_cols[2]

    # Mean rating score per condition across all participants
    human_cells = (
        df_human.groupby(
            ["modulation", "feature", "source", "rating_stimulus"], as_index=False
        )["rating_score"]
        .mean()
        .rename(columns={"rating_score": "distance"})
    )
    human_cells["loss_fn"] = "human"
    return human_cells


def compute_interaction_pooled_anova(
    df_data: pd.DataFrame,
    factors: list[str] | None = None,
    max_interaction: int = 2,
    dv: str = "distance",
) -> pd.DataFrame:
    """Compute interaction-pooled ANOVA and variance decomposition using statsmodels.

    ANOVA table is computed via statsmodels.api.stats.anova_lm (Type II SS).
    Effect sizes (np2, eta_sq) and multiple testing corrections (p_bonf, p_fdr)
    are calculated using vectorized operations equivalent to pingouin and statsmodels.stats.multitest.
    """
    if factors is None:
        factors = list(ALL_FACTORS)

    effective_max_interaction = min(max_interaction, len(factors) - 1)
    if effective_max_interaction < 1:
        raise ValueError(
            f"At least 2 factors are required for unreplicated interaction-pooled ANOVA; got {len(factors)}."
        )

    factor_terms = " + ".join(f"C({f})" for f in factors)
    if effective_max_interaction == 1:
        formula = f"{dv} ~ {factor_terms}"
    else:
        formula = f"{dv} ~ ({factor_terms})**{effective_max_interaction}"

    model = ols(formula, data=df_data).fit()

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

    # Degrees of Freedom as integer
    aov["DF"] = aov["DF"].astype(int)

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
    """Format ANOVA table for clean terminal display with adaptive precision."""
    df_disp = df.copy()

    def _fmt(val):
        if pd.isna(val):
            return ""
        if 0 < abs(val) < 0.0001:
            return f"{val:.2e}"
        return f"{val:.4f}"

    for col in [
        "SS",
        "MS",
        "F",
        "p_unc",
        "p_bonf",
        "p_fdr",
        "np2",
        "eta_sq",
        "pct_var",
    ]:
        if col in df_disp.columns:
            df_disp[col] = df_disp[col].apply(_fmt)

    df_disp["DF"] = df_disp["DF"].astype(int)
    return df_disp


def extract_variance_record(
    res_df: pd.DataFrame,
    label: str,
    factors: list[str] | None = None,
) -> dict[str, float | str]:
    """Extract variance decomposition percentages from an ANOVA result DataFrame."""
    if factors is None:
        factors = list(ALL_FACTORS)

    term_map = dict(zip(res_df["Source"], res_df["pct_var"]))
    rec: dict[str, float | str] = {"Metric": label}

    for factor in DISPLAY_FACTOR_ORDER:
        if factor in factors:
            disp_col = FACTOR_DISPLAY_NAMES[factor]
            rec[disp_col] = term_map.get(factor, 0.0)

    two_way_terms = [k for k in term_map if k.count(":") == 1]
    if two_way_terms:
        rec["% Var (2-Way Inter.)"] = sum(term_map[k] for k in two_way_terms)

    three_way_terms = [k for k in term_map if k.count(":") == 2]
    if three_way_terms:
        rec["% Var (3-Way Inter.)"] = sum(term_map[k] for k in three_way_terms)

    rec["% Var (Residual)"] = term_map.get("Residual (pooled)", 0.0)
    return rec


def run_variance_analysis(
    data_path: Path,
    human_data_path: Path | None = None,
    output_path: Path | None = None,
    loss_fns: list[str] | None = None,
    pool_factors: list[str] | None = None,
    max_interaction: int = 2,
):
    """Run interaction-pooled ANOVA and variance decomposition for loss functions & human ratings."""
    normalized_pool = normalize_pool_factors(pool_factors)
    remaining_factors = [f for f in ALL_FACTORS if f not in normalized_pool]
    effective_max_interaction = min(max_interaction, len(remaining_factors) - 1)

    if normalized_pool:
        log.info(
            f"Pooling ratings/distances across factor(s): {normalized_pool}. "
            f"Remaining ANOVA factors: {remaining_factors}"
        )
        if max_interaction > effective_max_interaction:
            log.info(
                f"Clamping max_interaction from {max_interaction} to {effective_max_interaction} "
                f"to ensure residual degrees of freedom."
            )

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
        f"Analyzing {len(selected_losses)} loss functions with max_interaction={effective_max_interaction} "
        f"across factors: {remaining_factors}..."
    )

    results_by_loss: dict[str, pd.DataFrame] = {}

    # 1. Process human listening study benchmark if available
    human_record: dict[str, float | str] | None = None
    if human_data_path and human_data_path.exists():
        log.info(f"Loading human listening benchmark data from: {human_data_path}")
        df_human = load_human_data(human_data_path)
        df_human_pooled = pool_data(df_human, remaining_factors)
        human_res = compute_interaction_pooled_anova(
            df_human_pooled,
            factors=remaining_factors,
            max_interaction=effective_max_interaction,
            dv="distance",
        )
        results_by_loss["human"] = human_res
        human_record = extract_variance_record(
            human_res,
            label="Human Listeners (Reference)",
            factors=remaining_factors,
        )

    # 2. Process algorithmic loss functions
    loss_summary_records = []
    for loss in selected_losses:
        sub = prepare_loss_data(df, loss)
        sub_pooled = pool_data(sub, remaining_factors)
        res_df = compute_interaction_pooled_anova(
            sub_pooled,
            factors=remaining_factors,
            max_interaction=effective_max_interaction,
            dv="distance",
        )
        results_by_loss[loss] = res_df
        loss_summary_records.append(
            extract_variance_record(
                res_df,
                label=loss,
                factors=remaining_factors,
            )
        )

    if normalized_pool:
        header_title = (
            f"INTERACTION-POOLED ANOVA & VARIANCE DECOMPOSITION "
            f"(POOLED OVER: {', '.join(normalized_pool).upper()} | "
            f"REMAINING: {', '.join(remaining_factors).upper()} | "
            f"Pooled Order: >{effective_max_interaction}-Way)"
        )
    else:
        header_title = (
            f"INTERACTION-POOLED ANOVA & VARIANCE DECOMPOSITION "
            f"(Pooled Order: >{effective_max_interaction}-Way)"
        )

    sep_bar = "=" * max(135, len(header_title))
    log.info("\n" + sep_bar)
    log.info(header_title)
    log.info(sep_bar)

    # Print human benchmark detailed ANOVA first if present
    if "human" in results_by_loss:
        log.info("\n--- Reference Metric: Human Listeners (Mean Rating Scores) ---")
        disp_human = format_table_for_display(results_by_loss["human"])
        log.info("\n" + disp_human.to_string(index=False))

    # Print loss functions detailed ANOVA
    for loss in selected_losses:
        log.info(f"\n--- Loss Function: {loss} ---")
        disp_df = format_table_for_display(results_by_loss[loss])
        log.info("\n" + disp_df.to_string(index=False))

    # Summary table: Sort loss functions by modulation amount sensitivity (% Var Amount) if present, else first metric
    summary_df = pd.DataFrame(loss_summary_records)
    sort_col = (
        "% Var (Amount)"
        if "% Var (Amount)" in summary_df.columns
        else [c for c in summary_df.columns if c != "Metric"][0]
    )
    summary_df = summary_df.sort_values(sort_col, ascending=False).reset_index(
        drop=True
    )

    log.info("\n" + sep_bar)
    log.info("SUMMARY: VARIANCE DECOMPOSITION PROFILES (% OF TOTAL VARIANCE EXPLAINED)")
    log.info(sep_bar)

    # Combine human reference at top followed by loss functions
    all_summary_rows = []
    if human_record:
        all_summary_rows.append(human_record)
    all_summary_rows.extend(summary_df.to_dict("records"))

    combined_summary_df = pd.DataFrame(all_summary_rows)
    disp_summary = combined_summary_df.copy()
    for col in disp_summary.columns[1:]:
        disp_summary[col] = disp_summary[col].apply(lambda x: f"{x:.2f}%")

    log.info("\n" + disp_summary.to_string(index=False))
    log.info(sep_bar + "\n")

    # 3. Export full results TSV (including human reference if present)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_records = []
        # Include human first
        if "human" in results_by_loss:
            df_h = results_by_loss["human"].copy()
            df_h.insert(0, "loss_fn", "human")
            export_records.append(df_h)
        for loss in selected_losses:
            df_copy = results_by_loss[loss].copy()
            df_copy.insert(0, "loss_fn", loss)
            export_records.append(df_copy)

        combined_df = pd.concat(export_records, ignore_index=True)
        combined_df.to_csv(output_path, sep="\t", index=False)
        log.info(
            f"Successfully saved detailed ANOVA & variance results to: {output_path}"
        )


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_data_path = repo_root / "data" / "distances.tsv"
    default_human_path = (
        repo_root / "data" / "listening_test_responses_postprocessed.tsv"
    )
    default_out_path = repo_root / "out" / "anova_variance_results.tsv"

    parser = argparse.ArgumentParser(
        description="Interaction-pooled ANOVA and variance decomposition for audio loss functions & human ratings."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=str(default_data_path),
        help=f"Path to distances TSV file (default: {default_data_path})",
    )
    parser.add_argument(
        "--human-data",
        default=str(default_human_path),
        help=f"Path to postprocessed human listening responses (default: {default_human_path})",
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
        "--pool-factors",
        "--pool",
        "--pool-over",
        nargs="+",
        default=None,
        # default=["source"],
        # default=["feature"],
        # default=["feature", "source"],
        help=(
            "Factor(s) to pool ratings / distances across before computing variance analysis "
            "(choices: 'feature', 'source', 'modulation', 'rating_stimulus'; aliases: 'amount', 'mod', 'feat', 'src'). "
            "For example: --pool-factors feature source"
        ),
    )
    parser.add_argument(
        "--max-interaction",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Maximum interaction order to include (default: 2; automatically clamped if fewer factors remain)",
    )
    args = parser.parse_args()

    run_variance_analysis(
        data_path=Path(args.data_path),
        human_data_path=Path(args.human_data) if args.human_data else None,
        output_path=Path(args.output) if args.output else None,
        loss_fns=args.loss_fn,
        pool_factors=args.pool_factors,
        max_interaction=args.max_interaction,
    )


if __name__ == "__main__":
    main()
