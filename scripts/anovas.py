"""ANOVA analysis for MUSHRA listening test data.

Converted from scripts/MushraDataAnalysis.R.
Computes repeated-measures ANOVAs and post-hoc pairwise tests using pingouin and statsmodels.
Assumes the input TSV/CSV dataset has already been cleaned and quality-filtered.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.formula._manager import FormulaManager
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.anova import _not_slice, _ssr_reduced_model

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def compute_rm_anova_with_effect_sizes(
    df: pd.DataFrame,
    dv: str,
    subject: str,
    within: list[str],
) -> pd.DataFrame:
    """Compute Repeated-Measures ANOVA for N within-subject factors with effect sizes.

    Computes:
    - Sums of Squares (SS) and Mean Squares (MS) for effects and errors
    - F-statistic and uncorrected p-value (p_unc)
    - Partial eta-squared (np2 = SS_effect / (SS_effect + SS_error))
    - Generalized eta-squared (ng2 = SS_effect / (SS_effect + SS_subject + sum(SS_errors)))
      following Olejnik & Algina (2003) and Bakeman (2005) for fully within-subject designs.
    """
    y = df[dv].values

    # Construct OLS endog and exog from string using patsy sum-to-zero contrasts
    within_terms = [f"C({i}, Sum)" for i in within]
    subject_term = f"C({subject}, Sum)"
    factors = [*within_terms, subject_term]
    mgr = FormulaManager()
    x = mgr.get_matrices("*".join(factors), data=df, pandas=False)
    term_slices = mgr.get_term_name_slices(x)
    for key in term_slices:
        ind = np.array([False] * x.shape[1])
        ind[term_slices[key]] = True
        term_slices[key] = np.array(ind)
    term_exclude = [":".join(factors)]
    ind = _not_slice(term_slices, term_exclude, x.shape[1])
    x = x[:, ind]

    # Fit full OLS model
    model = OLS(y, x)
    results = model.fit()
    if model.rank < x.shape[1]:
        raise ValueError("Independent variables are collinear.")
    for i in term_exclude:
        term_slices.pop(i)
    for key in term_slices:
        term_slices[key] = term_slices[key][ind]
    params = results.params
    df_resid = results.df_resid
    ssr = results.ssr

    # Calculate Subject Sum of Squares
    subj_key = subject_term
    ssr_subj, _ = _ssr_reduced_model(y, x, term_slices, params, [subj_key])
    ss_subject = ssr_subj - ssr

    records = []
    all_ss_error = 0.0
    for key in term_slices:
        if subject not in str(key) and str(key) not in ("Intercept", "1"):
            ssr1, df_resid1 = _ssr_reduced_model(y, x, term_slices, params, [key])
            df1 = df_resid1 - df_resid
            ss_effect = ssr1 - ssr
            msm = ss_effect / df1

            err_key = str(key) + ":" + subject_term
            if err_key in term_slices:
                ssr_err, df_err_res = _ssr_reduced_model(
                    y, x, term_slices, params, [err_key]
                )
                df2 = df_err_res - df_resid
                ss_error = ssr_err - ssr
                mse = ss_error / df2
            else:
                df2 = df_resid
                ss_error = ssr
                mse = ssr / df_resid

            all_ss_error += ss_error
            F = msm / mse if mse > 0 else np.nan
            p = stats.f.sf(F, df1, df2) if not np.isnan(F) else np.nan
            term = str(key).replace("C(", "").replace(", Sum)", "")
            records.append(
                {
                    "Source": term,
                    "SS": ss_effect,
                    "DF1": df1,
                    "DF2": df2,
                    "MS": msm,
                    "F": F,
                    "p_unc": p,
                    "ss_error": ss_error,
                }
            )

    table = pd.DataFrame(records)
    # Partial eta-squared: np2 = (F * DF1) / (F * DF1 + DF2)
    table["np2"] = (table["F"] * table["DF1"]) / (
        table["F"] * table["DF1"] + table["DF2"]
    )
    # Generalized eta-squared: ng2 = SS_effect / (SS_effect + SS_subject + sum(all_error_SS))
    denom_ges = ss_subject + all_ss_error
    table["ng2"] = table["SS"] / (table["SS"] + denom_ges)

    col_order = ["Source", "SS", "DF1", "DF2", "MS", "F", "p_unc", "np2", "ng2"]
    return table[col_order]


def _resolve_input_col(df: pd.DataFrame, dv: str, input_col: str | None = None) -> str:
    """Resolve the column to aggregate or measure.

    If input_col is explicitly specified, returns it (raising KeyError if missing).
    Otherwise, defaults to 'rating_score' if present in df.
    Otherwise, falls back to dv if present in df.
    Raises KeyError if neither is found in df.
    """
    if input_col is not None:
        if input_col not in df.columns:
            raise KeyError(
                f"Specified input column '{input_col}' not found in DataFrame."
            )
        return input_col
    if "rating_score" in df.columns:
        return "rating_score"
    if dv in df.columns:
        return dv
    raise KeyError(
        f"Neither default input column 'rating_score' nor dv '{dv}' found in DataFrame columns: {list(df.columns)}"
    )


def _filter_balanced_subjects(
    df: pd.DataFrame, subject_col: str, group_cols: list[str], dv: str
) -> pd.DataFrame:
    """Filter to subjects that have complete and balanced data across all within-subject cells.

    Verifies that each retained subject has exactly 1 valid observation per
    unique combination of group_cols across the full within-subject factorial design.
    """
    df_valid = df.dropna(subset=[*group_cols, dv])
    if df_valid.empty:
        return df_valid.copy()

    # Total expected cells across the full factorial combinations of group_cols
    expected_cells = int(np.prod([df_valid[col].nunique() for col in group_cols]))

    # Count observations per (subject, *group_cols) cell
    cell_obs = df_valid.groupby([subject_col, *group_cols])[dv].count()

    # A subject is balanced if:
    # 1. Exactly expected_cells unique combinations of group_cols are present
    # 2. Every combination has exactly 1 observation (no duplicate combinations)
    cell_sizes = cell_obs.groupby(level=subject_col).size()
    cell_all_ones = (cell_obs == 1).groupby(level=subject_col).all()

    balanced_mask = (cell_sizes == expected_cells) & cell_all_ones
    complete_subjects = balanced_mask[balanced_mask].index

    all_subjects = df[subject_col].dropna().unique()
    if len(complete_subjects) < len(all_subjects):
        log.info(
            f"Using {len(complete_subjects)} of {len(all_subjects)} subjects "
            f"with complete data for balanced design across {group_cols}."
        )
    return df_valid[df_valid[subject_col].isin(complete_subjects)].copy()


def compute_anova_4way_unpooled(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: str | None = None,
) -> pd.DataFrame:
    """4-way Repeated-Measures ANOVA (unpooled): modulation x feature x source x rating_stimulus (4 amounts).

    Uses all 4 individual stimulus condition ratings directly (DF1=3 for amount), without
    dichotomizing into Low vs High groups.
    Returns a DataFrame with F, p_unc, np2 (partial eta-squared), and ng2 (generalized eta-squared).
    """
    log.info(
        "Computing 4-way Repeated-Measures ANOVA (unpooled: modulation x feature x source x rating_stimulus)..."
    )
    in_col = _resolve_input_col(df, dv, input_col)
    mod_avg_4way_unpooled = (
        df.groupby(
            [subject, "modulation", "feature", "source", "rating_stimulus"],
            as_index=False,
        )[in_col]
        .mean()
        .rename(columns={in_col: dv})
    )

    within = ["modulation", "feature", "source", "rating_stimulus"]
    mod_avg_bal = _filter_balanced_subjects(mod_avg_4way_unpooled, subject, within, dv)

    # Save to .tsv for R compatibility if needed
    mod_avg_bal.to_csv("../out/4_way_data.tsv", sep="\t", index=False)

    return compute_rm_anova_with_effect_sizes(
        mod_avg_bal, dv=dv, subject=subject, within=within
    )


def compute_anova_4way_pooled(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: str | None = None,
) -> pd.DataFrame:
    """4-way Repeated-Measures ANOVA (pooled): modulation x feature x source x amount_group (Low vs High).

    Equivalent to R:
        anova_data_4way <- plot_data_grouped %>% group_by(session_uuid, modulation, feature, source, amount_group) ...
        anova_results_4way
    Returns a DataFrame with F, p_unc, np2 (partial eta-squared), and ng2 (generalized eta-squared).
    """
    log.info(
        "Computing 4-way Repeated-Measures ANOVA (pooled: modulation x feature x source x amount_group)..."
    )
    in_col = _resolve_input_col(df, dv, input_col)
    df_valid = df.dropna(subset=["amount_group"])
    mod_avg_4way = (
        df_valid.groupby(
            [subject, "modulation", "feature", "source", "amount_group"], as_index=False
        )[in_col]
        .mean()
        .rename(columns={in_col: dv})
    )

    within = ["modulation", "feature", "source", "amount_group"]
    mod_avg_bal = _filter_balanced_subjects(mod_avg_4way, subject, within, dv)

    return compute_rm_anova_with_effect_sizes(
        mod_avg_bal, dv=dv, subject=subject, within=within
    )


def compute_anova_3way(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: str | None = None,
) -> pd.DataFrame:
    """3-way Repeated-Measures ANOVA: modulation x feature x source.

    Equivalent to R:
        mod_avg %>% anova_test(dv = mean_rating, wid = session_uuid, within = c(modulation, feature, source))
    Returns a DataFrame with F, p_unc, np2 (partial eta-squared), and ng2 (generalized eta-squared).
    """
    log.info(
        "Computing 3-way Repeated-Measures ANOVA (modulation x feature x source)..."
    )
    in_col = _resolve_input_col(df, dv, input_col)
    mod_avg = (
        df.groupby(
            [subject, "modulation", "feature", "source"], as_index=False
        )[in_col]
        .mean()
        .rename(columns={in_col: dv})
    )

    within = ["modulation", "feature", "source"]
    mod_avg_bal = _filter_balanced_subjects(mod_avg, subject, within, dv)

    return compute_rm_anova_with_effect_sizes(
        mod_avg_bal, dv=dv, subject=subject, within=within
    )


def compute_anova_2way_modulation_feature(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2-way Repeated-Measures ANOVA: modulation x feature.

    Equivalent to R:
        mod_avg_2way %>% anova_test(dv = mean_rating, wid = session_uuid, within = c(modulation, feature))
    Computes ANOVA using both Pingouin (pg.rm_anova) and the custom helper function
    (compute_rm_anova_with_effect_sizes) for comparison.
    Returns:
        tuple of (aov_pingouin, aov_helper)
    """
    log.info("Computing 2-way Repeated-Measures ANOVA (modulation x feature)...")
    in_col = _resolve_input_col(df, dv, input_col)
    mod_avg_2way = (
        df.groupby([subject, "modulation", "feature"], as_index=False)[in_col]
        .mean()
        .rename(columns={in_col: dv})
    )

    within = ["modulation", "feature"]
    mod_avg_bal = _filter_balanced_subjects(mod_avg_2way, subject, within, dv)

    aov_pg = pg.rm_anova(
        data=mod_avg_bal,
        dv=dv,
        within=within,
        subject=subject,
        detailed=True,
    )
    aov_helper = compute_rm_anova_with_effect_sizes(
        mod_avg_bal, dv=dv, subject=subject, within=within
    )
    return aov_pg, aov_helper


def compute_anova_2way_modulation_amount(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2-way Repeated-Measures ANOVA: modulation x amount_group (Low vs High).

    Equivalent to R:
        anova_results_2way <- mod_avg_2way %>% anova_test(dv = mean_rating, wid = session_uuid, within = c(modulation, amount_group))
    Computes ANOVA using both Pingouin (pg.rm_anova) and the custom helper function
    (compute_rm_anova_with_effect_sizes) for comparison.
    Returns:
        tuple of (aov_pingouin, aov_helper)
    """
    log.info("Computing 2-way Repeated-Measures ANOVA (modulation x amount_group)...")
    in_col = _resolve_input_col(df, dv, input_col)
    df_valid = df.dropna(subset=["amount_group"])
    mod_avg_amount = (
        df_valid.groupby([subject, "modulation", "amount_group"], as_index=False)[
            in_col
        ]
        .mean()
        .rename(columns={in_col: dv})
    )

    within = ["modulation", "amount_group"]
    mod_avg_bal = _filter_balanced_subjects(mod_avg_amount, subject, within, dv)

    aov_pg = pg.rm_anova(
        data=mod_avg_bal,
        dv=dv,
        within=within,
        subject=subject,
        detailed=True,
    )
    aov_helper = compute_rm_anova_with_effect_sizes(
        mod_avg_bal, dv=dv, subject=subject, within=within
    )
    return aov_pg, aov_helper


def compute_pairwise_posthocs(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Pairwise post-hoc paired t-tests with Bonferroni correction.

    Equivalent to R:
        pairwise_t_test(mean_rating ~ modulation, paired = TRUE, p.adjust.method = "bonferroni")
        pairwise_t_test(mean_rating ~ feature, paired = TRUE, p.adjust.method = "bonferroni")
        pairwise_t_test(mean_rating ~ rating_stimulus, paired = TRUE, p.adjust.method = "bonferroni")
    """
    log.info("Computing pairwise post-hoc tests (Bonferroni adjusted)...")
    in_col = _resolve_input_col(df, dv, input_col)

    # 1. Modulation pairwise comparisons
    mod_means = (
        df.groupby([subject, "modulation"], as_index=False)[in_col]
        .mean()
        .rename(columns={in_col: dv})
    )
    mod_means_comp = _filter_balanced_subjects(mod_means, subject, ["modulation"], dv)

    pw_modulation = pg.pairwise_tests(
        data=mod_means_comp,
        dv=dv,
        within="modulation",
        subject=subject,
        padjust="bonf",
    )

    # 2. Feature pairwise comparisons
    feat_means = (
        df.groupby([subject, "feature"], as_index=False)[in_col]
        .mean()
        .rename(columns={in_col: dv})
    )
    feat_means_comp = _filter_balanced_subjects(feat_means, subject, ["feature"], dv)

    pw_feature = pg.pairwise_tests(
        data=feat_means_comp,
        dv=dv,
        within="feature",
        subject=subject,
        padjust="bonf",
    )

    # 3. Amount (four amounts / rating_stimulus) pairwise comparisons
    amount_col = (
        "rating_stimulus"
        if "rating_stimulus" in df.columns
        else ("amount" if "amount" in df.columns else None)
    )
    pw_amount = None
    if amount_col is not None:
        df_amount = (
            df[df[amount_col] != "reference"].copy()
            if "reference" in df[amount_col].values
            else df
        )
        amount_means = (
            df_amount.groupby([subject, amount_col], as_index=False)[in_col]
            .mean()
            .rename(columns={in_col: dv})
        )
        amount_means_comp = _filter_balanced_subjects(
            amount_means, subject, [amount_col], dv
        )

        pw_amount = pg.pairwise_tests(
            data=amount_means_comp,
            dv=dv,
            within=amount_col,
            subject=subject,
            padjust="bonf",
        )

    results = {
        "modulation": pw_modulation,
        "feature": pw_feature,
    }
    if pw_amount is not None:
        results["amount"] = pw_amount
        results["rating_stimulus"] = pw_amount

    return results


def compute_normality_tests(
    df: pd.DataFrame,
    group_by: list[str] | None = None,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: str | None = None,
) -> pd.DataFrame:
    """Shapiro-Wilk test for normality across factor combinations.

    Equivalent to R:
        shapiro.test(mean_rating)
    """
    if group_by is None:
        group_by = ["modulation", "feature", "source"]

    log.info(f"Computing Shapiro-Wilk normality tests grouped by {group_by}...")
    in_col = _resolve_input_col(df, dv, input_col)
    grouped = (
        df.groupby([subject] + group_by, as_index=False)[in_col]
        .mean()
        .rename(columns={in_col: dv})
    )

    records = []
    for keys, group in grouped.groupby(group_by):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ratings = group[dv].dropna()
        if len(ratings) >= 3:
            w_stat, p_val = stats.shapiro(ratings)
        else:
            w_stat, p_val = float("nan"), float("nan")

        rec = dict(zip(group_by, keys))
        rec["n"] = len(ratings)
        rec["W"] = w_stat
        rec["p"] = p_val
        rec["normality"] = "Normal" if p_val > 0.05 else "Non-normal"
        records.append(rec)

    return pd.DataFrame(records)


def run_all_anovas(
    data_source: Union[str, Path, pd.DataFrame],
    dv: str = "mean_rating",
    input_col: str | None = None,
):
    """Execute all ANOVA analyses from MushraDataAnalysis.R and print summaries.

    Assumes the input TSV/CSV dataset has already been cleaned and quality-filtered.
    Extracts within-subject factors (modulation, feature, source, amount_group) if not already present.
    """
    if isinstance(data_source, (str, Path)):
        file_path = Path(data_source).expanduser().resolve()
        log.info(f"Loading prepared data from {file_path}")
        sep = "\t" if file_path.suffix == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep)
    else:
        df = data_source.copy()

    # Exclude reference stimulus ratings for ANOVA if present
    if "rating_stimulus" in df.columns:
        df = df[df["rating_stimulus"] != "reference"].copy()

    if "trial_id" in df.columns:
        # Separate trial_id into modulation, feature, source if not already present
        if "modulation" not in df.columns:
            split_cols = df["trial_id"].str.split("_", expand=True)
            if split_cols.shape[1] >= 3:
                df["modulation"] = split_cols[0]
                df["feature"] = split_cols[1]
                df["source"] = split_cols[2]

    # Map conditions to amount_group: Low (conditions a/b) vs High (conditions c/d) if not already present
    if "amount_group" not in df.columns and "rating_stimulus" in df.columns:
        cond_map = {
            "condition_a": "Low",
            "condition_b": "Low",
            "condition_c": "High",
            "condition_d": "High",
        }
        df["amount_group"] = df["rating_stimulus"].map(cond_map)

    log.info("\n" + "=" * 65)
    log.info(
        " 1. FOUR-WAY REPEATED MEASURES ANOVA (modulation x feature x source x rating_stimulus [UNPOOLED])"
    )
    log.info("=" * 65)
    aov_4way_unpooled = compute_anova_4way_unpooled(df, dv=dv, input_col=input_col)
    log.info("\n" + aov_4way_unpooled.to_string(index=False))

    log.info("\n" + "=" * 65)
    log.info(
        " 2. FOUR-WAY REPEATED MEASURES ANOVA (modulation x feature x source x amount_group [POOLED])"
    )
    log.info("=" * 65)
    aov_4way = compute_anova_4way_pooled(df, dv=dv, input_col=input_col)
    log.info("\n" + aov_4way.to_string(index=False))

    log.info("\n" + "=" * 65)
    log.info(" 3. THREE-WAY REPEATED MEASURES ANOVA (modulation x feature x source)")
    log.info("=" * 65)
    aov_3way = compute_anova_3way(df, dv=dv, input_col=input_col)
    log.info("\n" + aov_3way.to_string(index=False))

    log.info("\n" + "=" * 65)
    log.info(" 4. TWO-WAY REPEATED MEASURES ANOVA (modulation x feature)")
    log.info("=" * 65)
    aov_2way_mf_pg, aov_2way_mf_helper = compute_anova_2way_modulation_feature(
        df, dv=dv, input_col=input_col
    )
    log.info(
        "\n[Pingouin (pg.rm_anova, detailed=True)]:\n"
        + aov_2way_mf_pg.to_string(index=False)
    )
    log.info(
        "\n[Helper Function (compute_rm_anova_with_effect_sizes)]:\n"
        + aov_2way_mf_helper.to_string(index=False)
    )

    log.info("\n" + "=" * 65)
    log.info(" 5. TWO-WAY REPEATED MEASURES ANOVA (modulation x amount_group)")
    log.info("=" * 65)
    aov_2way_ma_pg, aov_2way_ma_helper = compute_anova_2way_modulation_amount(
        df, dv=dv, input_col=input_col
    )
    log.info(
        "\n[Pingouin (pg.rm_anova, detailed=True)]:\n"
        + aov_2way_ma_pg.to_string(index=False)
    )
    log.info(
        "\n[Helper Function (compute_rm_anova_with_effect_sizes)]:\n"
        + aov_2way_ma_helper.to_string(index=False)
    )

    log.info("\n" + "=" * 65)
    log.info(" 6. POST-HOC PAIRWISE TESTS (BONFERRONI)")
    log.info("=" * 65)
    posthocs = compute_pairwise_posthocs(df, dv=dv, input_col=input_col)
    log.info("\n[Post-hoc: Modulation]")
    log.info(posthocs["modulation"].to_string(index=False))
    log.info("\n[Post-hoc: Feature]")
    log.info(posthocs["feature"].to_string(index=False))
    if "amount" in posthocs:
        log.info("\n[Post-hoc: Amount (4 conditions)]")
        log.info(posthocs["amount"].to_string(index=False))


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    default_data_path = (
        repo_root / "data" / "listening_test_responses_postprocessed.tsv"
    )

    parser = argparse.ArgumentParser(
        description="Run ANOVA analyses on prepared MUSHRA listening test data."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=str(default_data_path),
        help=f"Path to prepared MUSHRA data file (tsv or csv; default: {default_data_path})",
    )
    parser.add_argument(
        "--dv",
        default="mean_rating",
        help="Dependent variable column name for ANOVA (default: mean_rating)",
    )
    parser.add_argument(
        "--input-col",
        default=None,
        help="Input rating column to aggregate (default: rating_score if present, else dv)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        log.error(f"Data file not found at: {args.data_path}")
        parser.print_help()
        sys.exit(1)

    run_all_anovas(args.data_path, dv=args.dv, input_col=args.input_col)
