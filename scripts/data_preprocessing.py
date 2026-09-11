"""MUSHRA listening test data cleaning and preprocessing.

Converted from and aligned with scripts/MushraDataAnalysis.R.
Implements data quality heuristics:
1. Filters out training trials ('trial_id != "training"').
2. Filters out pilot/debugging sessions ('comments == "christhetree"').
3. Identifies and excludes bad trials per participant based on:
   - reference_score > 25 (hidden reference rated too high for difference rating)
   - all_identical (participant gave identical ratings to all stimuli)
   - total_time < 24000 ms (trial completed in less than 24 seconds)
   - rating_range < 10 (difference between max and min ratings is under 10)
4. Anti-joins to remove bad trials.
5. Saves the clean filtered dataset to TSV and outputs a detailed removal breakdown.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Union

import pandas as pd

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
    3. Identify bad trials per participant:
       - reference_score > 25 (hidden reference rated too high for difference rating)
       - all_identical: participant gave the exact same rating to all stimuli in trial
       - total_time < 24000 ms: trial completed in less than 24 seconds (rushed)
       - rating_range < 10: difference between max and min ratings is under 10
    4. Anti-join to remove all rows associated with bad trials.

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

    # 2. Identify bad trials (grouped by session_uuid and trial_id)
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

    crit_ref = (stats_df["reference_score"] > 25).fillna(False)
    crit_identical = stats_df["all_identical"]
    crit_time = (
        stats_df["total_time"] < 24000
        if "total_time" in stats_df.columns
        else pd.Series(False, index=stats_df.index)
    )
    crit_range = stats_df["rating_range"] < 10

    bad_mask = crit_ref | crit_identical | crit_time | crit_range

    bad_trials = stats_df[bad_mask].reset_index()[["session_uuid", "trial_id"]].copy()
    bad_trials.attrs["criteria_counts"] = {
        "reference_score > 25": int(crit_ref.sum()),
        "total_time < 24000 ms": int(crit_time.sum()),
        "rating_range < 10": int(crit_range.sum()),
        "all_identical": int(crit_identical.sum()),
    }
    log.info(f"Identified {len(bad_trials)} bad trials to exclude.")

    # 3. Anti-join: remove bad trials
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


def filter_and_save(
    input_path: Union[str, Path],
    output_path: Union[str, Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Filter raw MUSHRA listening test data, save the filtered dataset to TSV, and report removals.

    Prints granular breakdown of trials removed across the quality criteria and count of complete users.
    """
    input_path = Path(input_path).expanduser().resolve()
    sep = "\t" if input_path.suffix == ".tsv" else ","
    df_raw = pd.read_csv(input_path, sep=sep)

    initial_users = df_raw["session_uuid"].nunique() if "session_uuid" in df_raw.columns else 0
    initial_trials = (
        df_raw[["session_uuid", "trial_id"]].drop_duplicates().shape[0]
        if "session_uuid" in df_raw.columns and "trial_id" in df_raw.columns
        else 0
    )
    initial_user_ids = set(df_raw["session_uuid"]) if "session_uuid" in df_raw.columns else set()

    data_filtered, bad_trials = filter_mushra_data(df_raw)

    final_users = (
        data_filtered["session_uuid"].nunique() if "session_uuid" in data_filtered.columns else 0
    )
    final_trials = (
        data_filtered[["session_uuid", "trial_id"]].drop_duplicates().shape[0]
        if "session_uuid" in data_filtered.columns and "trial_id" in data_filtered.columns
        else 0
    )
    final_user_ids = (
        set(data_filtered["session_uuid"]) if "session_uuid" in data_filtered.columns else set()
    )

    trials_removed = initial_trials - final_trials
    users_removed = initial_users - final_users
    removed_user_ids = sorted(list(initial_user_ids - final_user_ids))

    # Calculate complete users (18 trials)
    trials_per_user = (
        data_filtered.groupby("session_uuid")["trial_id"].nunique()
        if "session_uuid" in data_filtered.columns and "trial_id" in data_filtered.columns
        else pd.Series(dtype=int)
    )
    complete_users = int((trials_per_user == 18).sum())

    criteria_counts = bad_trials.attrs.get("criteria_counts", {})

    if output_path is None:
        stem = input_path.stem
        output_name = f"{stem}_filtered.tsv"
        out_file = input_path.parent / output_name
    else:
        out_file = Path(output_path).expanduser().resolve()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    data_filtered.to_csv(out_file, sep="\t", index=False)

    print("=" * 65)
    print("DATA FILTERING SUMMARY")
    print("=" * 65)
    print(f"Filtered data saved to: {out_file}\n")
    print(f"Trials removed: {trials_removed}")
    training_removed = initial_trials - len(bad_trials) - final_trials
    if training_removed > 0:
        print(f"  - Training/pilot trials excluded: {training_removed}")
    print(f"  - Bad quality trials excluded:    {len(bad_trials)}")
    if criteria_counts:
        print("    Breakdown by criterion (trials may match multiple):")
        for criterion, count in criteria_counts.items():
            print(f"      * {criterion:<22}: {count} trials")
    print(f"\nUsers removed:  {users_removed}")
    if removed_user_ids:
        print(f"Removed user session UUID(s): {', '.join(removed_user_ids)}")
    print(f"\nFinal dataset:")
    print(f"  - Initial dataset: {initial_trials} trials across {initial_users} users ({len(df_raw)} rows)")
    print(f"  - Filtered dataset: {final_trials} trials across {final_users} users ({len(data_filtered)} rows)")
    pct_complete = (complete_users / final_users * 100) if final_users > 0 else 0.0
    print(f"  - Users with complete data (18 trials): {complete_users} of {final_users} ({pct_complete:.1f}%)")
    print("=" * 65)

    return data_filtered, bad_trials, out_file


if __name__ == "__main__":
    default_data_path = (
        # Path(__file__).resolve().parent.parent / "data" / "listening_test_responses_device_filtered.tsv"
        Path(__file__).resolve().parent.parent / "data" / "listening_test_responses_device_filtered_filtered.tsv"
    )

    parser = argparse.ArgumentParser(
        description="Preprocess and quality-filter MUSHRA listening test data."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=str(default_data_path),
        help=f"Path to input MUSHRA data file (tsv or csv; default: {default_data_path})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to save the filtered TSV file (default: data/listening_test_responses_filtered.tsv)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        log.error(f"Data file not found at: {args.data_path}")
        parser.print_help()
        sys.exit(1)

    filter_and_save(args.data_path, output_path=args.output)
