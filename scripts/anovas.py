"""ANOVA analysis for MUSHRA listening test data.

Converted from scripts/MushraDataAnalysis.R.
Includes data filtering (unsuitable devices, bad trials, reference score thresholds)
and computes repeated-measures ANOVAs and post-hoc pairwise tests using pingouin and statsmodels.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Union

import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.stats.anova import AnovaRM

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def filter_mushra_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply data cleaning and filtering heuristics from MushraDataAnalysis.R.

    Filtering steps:
    1. Filter out training trials ('trial_id != "training"').
    2. Filter out pilot/debugging entries where comments == 'christhetree'.
    4. Identify bad trials per participant:
       - reference_score > 25 (hidden reference rated too high for difference rating)
       - all_identical: participant gave the exact same rating to all stimuli in trial
       - total_time < 24000 ms: trial completed in less than 24 seconds (rushed)
       - rating_range < 10: difference between max and min ratings is under 10
    5. Anti-join to remove all rows associated with bad trials.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (data_filtered, bad_trials)
    """
    initial_rows = len(df)
    initial_participants = df["session_uuid"].nunique() if "session_uuid" in df.columns else 0
    log.info(f"Starting data filtering: {initial_rows} rows from {initial_participants} participants.")

    # 1. Basic setup: drop training trials and pilot comments
    clean_mask = pd.Series(True, index=df.index)
    if "trial_id" in df.columns:
        clean_mask &= df["trial_id"] != "training"

    if "comments" in df.columns:
        is_christhetree = (
            df["comments"].dropna().astype(str).str.strip().str.lower() == "christhetree"
        )
        clean_mask &= ~df.index.isin(is_christhetree[is_christhetree].index)

    data_clean = df[clean_mask].copy()

    # 3. Identify bad trials (grouped by session_uuid and trial_id)
    if not all(col in data_clean.columns for col in ["session_uuid", "trial_id", "rating_score"]):
        log.warning("Required columns for bad trial detection not found; skipping trial-level filtering.")
        return data_clean, pd.DataFrame()

    ref_scores = (
        data_clean[data_clean.get("rating_stimulus", pd.Series(index=data_clean.index)) == "reference"]
        .groupby(["session_uuid", "trial_id"])["rating_score"]
        .first()
        .rename("reference_score")
    )

    agg_dict = {
        "n_distinct": ("rating_score", "nunique"),
        "rating_min": ("rating_score", "min"),
        "rating_max": ("rating_score", "max"),
    }
    if "rating_time" in data_clean.columns:
        agg_dict["total_time"] = ("rating_time", "max")

    stats_df = data_clean.groupby(["session_uuid", "trial_id"]).agg(**agg_dict).join(ref_scores)

    stats_df["all_identical"] = stats_df["n_distinct"] == 1
    stats_df["rating_range"] = stats_df["rating_max"] - stats_df["rating_min"]

    bad_mask = (
        (stats_df["reference_score"] > 25)
        | (stats_df["all_identical"])
        | (stats_df["rating_range"] < 10)
    )
    if "total_time" in stats_df.columns:
        bad_mask |= stats_df["total_time"] < 24000

    bad_trials = stats_df[bad_mask].reset_index()[["session_uuid", "trial_id"]]
    log.info(f"Identified {len(bad_trials)} bad trials to exclude.")

    # 4. Anti-join: remove bad trials
    data_filtered = data_clean.merge(
        bad_trials,
        on=["session_uuid", "trial_id"],
        how="left",
        indicator=True,
    )
    data_filtered = data_filtered[data_filtered["_merge"] == "left_only"].drop(columns=["_merge"])

    final_participants = data_filtered["session_uuid"].nunique()
    log.info(
        f"Filtered dataset ready: {len(data_filtered)} rows from {final_participants} participants."
    )
    return data_filtered, bad_trials


def prepare_mushra_data(
    data_source: Union[str, Path, pd.DataFrame],
    apply_filtering: bool = True,
) -> pd.DataFrame:
    """Load and prepare MUSHRA data for ANOVA analyses.

    Optionally applies the cleaning and filtering steps from MushraDataAnalysis.R.
    Extracts within-subject factor levels from `trial_id` (modulation, feature, source)
    and assigns `amount_group` ('Low' vs 'High') based on stimulus condition.
    """
    if isinstance(data_source, (str, Path)):
        file_path = Path(data_source).expanduser()
        log.info(f"Loading data from {file_path}")
        # Detect delimiter (tsv vs csv)
        sep = "\t" if file_path.suffix == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep)
    else:
        df = data_source.copy()

    # Apply data filtering if requested
    if apply_filtering:
        df, _ = filter_mushra_data(df)

    # Exclude reference stimulus ratings for ANOVA
    if "rating_stimulus" in df.columns:
        df = df[df["rating_stimulus"] != "reference"].copy()

    # Drop training trials if not already dropped
    if "trial_id" in df.columns:
        df = df[df["trial_id"] != "training"].copy()
        # Separate trial_id into modulation, feature, source (e.g. 'amp_brightness_real')
        split_cols = df["trial_id"].str.split("_", expand=True)
        if split_cols.shape[1] >= 3:
            df["modulation"] = split_cols[0]
            df["feature"] = split_cols[1]
            df["source"] = split_cols[2]

    # Map conditions to amount_group: Low (conditions a/b) vs High (conditions c/d)
    if "amount_group" not in df.columns and "rating_stimulus" in df.columns:
        cond_map = {
            "condition_a": "Low",
            "condition_b": "Low",
            "condition_c": "High",
            "condition_d": "High",
        }
        df["amount_group"] = df["rating_stimulus"].map(cond_map)

    return df


def _filter_balanced_subjects(
    df: pd.DataFrame, subject_col: str, group_cols: list[str], dv: str
) -> pd.DataFrame:
    """Filter to subjects that have complete data across all within-subject cells."""
    cell_counts = df.groupby(subject_col)[dv].count()
    expected_cells = 1
    for col in group_cols:
        expected_cells *= df[col].nunique()

    complete_subjects = cell_counts[cell_counts == expected_cells].index
    if len(complete_subjects) < len(cell_counts):
        log.info(
            f"Using {len(complete_subjects)} of {len(cell_counts)} subjects "
            f"with complete data for balanced design across {group_cols}."
        )
    return df[df[subject_col].isin(complete_subjects)].copy()


def compute_anova_3way(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
) -> AnovaRM:
    """3-way Repeated-Measures ANOVA: modulation x feature x source.

    Equivalent to R:
        mod_avg %>% anova_test(dv = mean_rating, wid = session_uuid, within = c(modulation, feature, source))
    """
    log.info("Computing 3-way Repeated-Measures ANOVA (modulation x feature x source)...")
    mod_avg = (
        df.groupby([subject, "trial_id", "modulation", "feature", "source"], as_index=False)[
            "rating_score"
        ]
        .mean()
        .rename(columns={"rating_score": dv})
    )

    within = ["modulation", "feature", "source"]
    mod_avg_bal = _filter_balanced_subjects(mod_avg, subject, within, dv)

    aov_3way = AnovaRM(mod_avg_bal, depvar=dv, subject=subject, within=within).fit()
    return aov_3way


def compute_anova_2way_modulation_feature(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
) -> pd.DataFrame:
    """2-way Repeated-Measures ANOVA: modulation x feature.

    Equivalent to R:
        mod_avg_2way %>% anova_test(dv = mean_rating, wid = session_uuid, within = c(modulation, feature))
    """
    log.info("Computing 2-way Repeated-Measures ANOVA (modulation x feature)...")
    mod_avg_2way = (
        df.groupby([subject, "modulation", "feature"], as_index=False)["rating_score"]
        .mean()
        .rename(columns={"rating_score": dv})
    )

    within = ["modulation", "feature"]
    mod_avg_bal = _filter_balanced_subjects(mod_avg_2way, subject, within, dv)

    aov_2way = pg.rm_anova(
        data=mod_avg_bal,
        dv=dv,
        within=within,
        subject=subject,
        detailed=True,
    )
    return aov_2way


def compute_anova_2way_modulation_amount(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
) -> pd.DataFrame:
    """2-way Repeated-Measures ANOVA: modulation x amount_group (Low vs High).

    Equivalent to R:
        anova_results_2way <- mod_avg_2way %>% anova_test(dv = mean_rating, wid = session_uuid, within = c(modulation, amount_group))
    """
    log.info("Computing 2-way Repeated-Measures ANOVA (modulation x amount_group)...")
    df_valid = df.dropna(subset=["amount_group"])
    mod_avg_amount = (
        df_valid.groupby([subject, "modulation", "amount_group"], as_index=False)[
            "rating_score"
        ]
        .mean()
        .rename(columns={"rating_score": dv})
    )

    within = ["modulation", "amount_group"]
    mod_avg_bal = _filter_balanced_subjects(mod_avg_amount, subject, within, dv)

    aov_amount = pg.rm_anova(
        data=mod_avg_bal,
        dv=dv,
        within=within,
        subject=subject,
        detailed=True,
    )
    return aov_amount


def compute_anova_4way(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
) -> AnovaRM:
    """4-way Repeated-Measures ANOVA: modulation x feature x source x amount_group.

    Equivalent to R:
        anova_data_4way <- plot_data_grouped %>% group_by(session_uuid, modulation, feature, source, amount_group) ...
        anova_results_4way
    """
    log.info(
        "Computing 4-way Repeated-Measures ANOVA (modulation x feature x source x amount_group)..."
    )
    df_valid = df.dropna(subset=["amount_group"])
    mod_avg_4way = (
        df_valid.groupby(
            [subject, "modulation", "feature", "source", "amount_group"], as_index=False
        )["rating_score"]
        .mean()
        .rename(columns={"rating_score": dv})
    )

    within = ["modulation", "feature", "source", "amount_group"]
    mod_avg_bal = _filter_balanced_subjects(mod_avg_4way, subject, within, dv)

    aov_4way = AnovaRM(mod_avg_bal, depvar=dv, subject=subject, within=within).fit()
    return aov_4way


def compute_pairwise_posthocs(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
) -> dict[str, pd.DataFrame]:
    """Pairwise post-hoc paired t-tests with Bonferroni correction.

    Equivalent to R:
        pairwise_t_test(mean_rating ~ modulation, paired = TRUE, p.adjust.method = "bonferroni")
        pairwise_t_test(mean_rating ~ feature, paired = TRUE, p.adjust.method = "bonferroni")
    """
    log.info("Computing pairwise post-hoc tests (Bonferroni adjusted)...")

    # 1. Modulation pairwise comparisons
    mod_means = (
        df.groupby([subject, "modulation"], as_index=False)["rating_score"]
        .mean()
        .rename(columns={"rating_score": dv})
    )
    comp_subs_mod = mod_means.groupby(subject)["modulation"].nunique()
    n_mods = mod_means["modulation"].nunique()
    mod_means_comp = mod_means[
        mod_means[subject].isin(comp_subs_mod[comp_subs_mod == n_mods].index)
    ]

    pw_modulation = pg.pairwise_tests(
        data=mod_means_comp,
        dv=dv,
        within="modulation",
        subject=subject,
        padjust="bonf",
    )

    # 2. Feature pairwise comparisons
    feat_means = (
        df.groupby([subject, "feature"], as_index=False)["rating_score"]
        .mean()
        .rename(columns={"rating_score": dv})
    )
    comp_subs_feat = feat_means.groupby(subject)["feature"].nunique()
    n_feats = feat_means["feature"].nunique()
    feat_means_comp = feat_means[
        feat_means[subject].isin(comp_subs_feat[comp_subs_feat == n_feats].index)
    ]

    pw_feature = pg.pairwise_tests(
        data=feat_means_comp,
        dv=dv,
        within="feature",
        subject=subject,
        padjust="bonf",
    )

    return {
        "modulation": pw_modulation,
        "feature": pw_feature,
    }


def compute_normality_tests(
    df: pd.DataFrame,
    group_by: list[str] | None = None,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
) -> pd.DataFrame:
    """Shapiro-Wilk test for normality across factor combinations.

    Equivalent to R:
        shapiro.test(mean_rating)
    """
    if group_by is None:
        group_by = ["modulation", "feature", "source"]

    log.info(f"Computing Shapiro-Wilk normality tests grouped by {group_by}...")
    grouped = (
        df.groupby([subject] + group_by, as_index=False)["rating_score"]
        .mean()
        .rename(columns={"rating_score": dv})
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
    apply_filtering: bool = True,
):
    """Execute all ANOVA analyses from MushraDataAnalysis.R and print summaries."""
    df = prepare_mushra_data(data_source, apply_filtering=apply_filtering)

    log.info("\n" + "=" * 65)
    log.info(" 1. THREE-WAY REPEATED MEASURES ANOVA (modulation x feature x source)")
    log.info("=" * 65)
    aov_3way = compute_anova_3way(df)
    log.info(aov_3way)

    log.info("\n" + "=" * 65)
    log.info(" 2. TWO-WAY REPEATED MEASURES ANOVA (modulation x feature)")
    log.info("=" * 65)
    aov_2way_mf = compute_anova_2way_modulation_feature(df)
    log.info(aov_2way_mf.to_string(index=False))

    log.info("\n" + "=" * 65)
    log.info(" 3. TWO-WAY REPEATED MEASURES ANOVA (modulation x amount_group)")
    log.info("=" * 65)
    aov_2way_ma = compute_anova_2way_modulation_amount(df)
    log.info(aov_2way_ma.to_string(index=False))

    log.info("\n" + "=" * 65)
    log.info(
        " 4. FOUR-WAY REPEATED MEASURES ANOVA (modulation x feature x source x amount_group)"
    )
    log.info("=" * 65)
    aov_4way = compute_anova_4way(df)
    log.info(aov_4way)

    log.info("\n" + "=" * 65)
    log.info(" 5. POST-HOC PAIRWISE TESTS (BONFERRONI)")
    log.info("=" * 65)
    posthocs = compute_pairwise_posthocs(df)
    log.info("\n[Post-hoc: Modulation]")
    log.info(posthocs["modulation"].to_string(index=False))
    log.info("\n[Post-hoc: Feature]")
    log.info(posthocs["feature"].to_string(index=False))


if __name__ == "__main__":
    default_data_path = (
        # Path(__file__).resolve().parent.parent / "data" / "listening_test_responses_device_filtered.tsv"
        Path(__file__).resolve().parent.parent / "data" / "listening_test_responses_2_participants.tsv"
    )

    parser = argparse.ArgumentParser(
        description="Run ANOVA analyses on MUSHRA listening test data."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=str(default_data_path),
        help=f"Path to the MUSHRA data file (tsv or csv; default: {default_data_path})",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable quality filtering (unsuitable devices, bad trials, etc.)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        log.error(f"Data file not found at: {args.data_path}")
        parser.print_help()
    else:
        run_all_anovas(args.data_path, apply_filtering=not args.no_filter)
