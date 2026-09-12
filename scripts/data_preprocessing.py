"""MUSHRA listening test data cleaning and preprocessing.

Converted from and aligned with scripts/MushraDataAnalysis.R.
Implements data quality heuristics:
1. Filters out training trials ('trial_id != "training"').
2. Filters out unreliable participants (user-level post-screening):
   - Removes all responses from users that rate the hidden reference higher than 10
     more than 15% of the time.
3. Identifies and excludes bad trials among remaining participants based on:
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
    ref_score_threshold: float = 10.0,
    ref_rate_threshold: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply data cleaning and filtering heuristics for MUSHRA listening test responses.

    Filtering steps:
    1. Filter out training trials ('trial_id != "training"').
    2. Filter out bad users (user-level post-screening):
       - Remove all responses from participants who rate the hidden reference
         higher than `ref_score_threshold` (10) more than `ref_rate_threshold` (15%) of the time.
    3. Identify bad trials among remaining participants:
       - all_identical: participant gave the exact same rating to all stimuli in trial
       - total_time < 24000 ms: trial completed in less than 24 seconds (rushed)
       - rating_range < 10: difference between max and min ratings is under 10
    4. Anti-join to remove all rows associated with bad trials.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (data_filtered, bad_trials)
    """
    initial_rows = len(df)
    initial_participants = (
        df["session_uuid"].nunique() if "session_uuid" in df.columns else 0
    )
    log.info(
        f"Starting data filtering: {initial_rows} rows from {initial_participants} participants."
    )

    # 1. Basic setup: drop training trials
    clean_mask = pd.Series(True, index=df.index)
    if "trial_id" in df.columns:
        clean_mask &= df["trial_id"] != "training"

    data_clean = df[clean_mask].copy()

    # 2. Filter out bad users (rated reference > ref_score_threshold in > ref_rate_threshold of trials)
    bad_users_list: list[dict] = []
    if "session_uuid" in data_clean.columns and "rating_score" in data_clean.columns:
        ref_rows = data_clean[
            data_clean.get("rating_stimulus", pd.Series(index=data_clean.index))
            == "reference"
        ]
        if not ref_rows.empty:
            user_ref_stats = (
                ref_rows.groupby("session_uuid")["rating_score"]
                .agg(
                    n_ref_trials="count",
                    n_gt_threshold=lambda s: int((s > ref_score_threshold).sum()),
                    pct_gt_threshold=lambda s: float((s > ref_score_threshold).mean()),
                )
                .reset_index()
            )
            bad_users_df = user_ref_stats[
                user_ref_stats["pct_gt_threshold"] > ref_rate_threshold
            ]
            bad_user_ids = set(bad_users_df["session_uuid"])
            bad_users_list = bad_users_df.to_dict("records")
            log.info(
                f"Identified {len(bad_user_ids)} bad users (rated ref > {ref_score_threshold} in > {ref_rate_threshold * 100:.0f}% of trials)."
            )
            data_clean = data_clean[
                ~data_clean["session_uuid"].isin(bad_user_ids)
            ].copy()

    # 3. Identify bad trials among remaining participants (grouped by session_uuid and trial_id)
    if not all(
        col in data_clean.columns
        for col in ["session_uuid", "trial_id", "rating_score"]
    ):
        log.warning(
            "Required columns for bad trial detection not found; skipping trial-level filtering."
        )
        bad_trials = pd.DataFrame()
        bad_trials.attrs["bad_users"] = bad_users_list
        bad_trials.attrs["criteria_counts"] = {}
        return data_clean, bad_trials

    agg_dict = {
        "n_distinct": ("rating_score", "nunique"),
        "rating_min": ("rating_score", "min"),
        "rating_max": ("rating_score", "max"),
    }
    if "rating_time" in data_clean.columns:
        agg_dict["total_time"] = ("rating_time", "max")

    stats_df = data_clean.groupby(["session_uuid", "trial_id"]).agg(**agg_dict)

    stats_df["all_identical"] = stats_df["n_distinct"] == 1
    stats_df["rating_range"] = stats_df["rating_max"] - stats_df["rating_min"]

    crit_identical = stats_df["all_identical"]
    crit_time = (
        stats_df["total_time"] < 24000
        if "total_time" in stats_df.columns
        else pd.Series(False, index=stats_df.index)
    )
    crit_range = stats_df["rating_range"] < 10

    bad_mask = crit_identical | crit_time | crit_range

    bad_trials = stats_df[bad_mask].reset_index()[["session_uuid", "trial_id"]].copy()
    bad_trials.attrs["bad_users"] = bad_users_list
    bad_trials.attrs["criteria_counts"] = {
        "total_time < 24000 ms": int(crit_time.sum()),
        "rating_range < 10": int(crit_range.sum()),
        "all_identical": int(crit_identical.sum()),
    }
    log.info(
        f"Identified {len(bad_trials)} bad trials to exclude among remaining participants."
    )

    # 4. Anti-join: remove bad trials
    data_filtered = data_clean.merge(
        bad_trials,
        on=["session_uuid", "trial_id"],
        how="left",
        indicator=True,
    )
    data_filtered = data_filtered[data_filtered["_merge"] == "left_only"].drop(
        columns=["_merge"]
    )

    final_participants = data_filtered["session_uuid"].nunique()
    log.info(
        f"Filtered dataset ready: {len(data_filtered)} rows from {final_participants} participants."
    )
    return data_filtered, bad_trials


def filter_and_save(
    input_path: Union[str, Path],
    output_path: Union[str, Path] | None = None,
    ref_score_threshold: float = 10.0,
    ref_rate_threshold: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Filter raw MUSHRA listening test data, save the filtered dataset to TSV, and report removals.

    Prints granular breakdown of users and trials removed across quality criteria and count of complete users.
    """
    input_path = Path(input_path).expanduser().resolve()
    sep = "\t" if input_path.suffix == ".tsv" else ","
    df_raw = pd.read_csv(input_path, sep=sep)

    initial_users = (
        df_raw["session_uuid"].nunique() if "session_uuid" in df_raw.columns else 0
    )
    initial_trials = (
        df_raw[["session_uuid", "trial_id"]].drop_duplicates().shape[0]
        if "session_uuid" in df_raw.columns and "trial_id" in df_raw.columns
        else 0
    )
    initial_user_ids = (
        set(df_raw["session_uuid"]) if "session_uuid" in df_raw.columns else set()
    )

    data_filtered, bad_trials = filter_mushra_data(
        df_raw,
        ref_score_threshold=ref_score_threshold,
        ref_rate_threshold=ref_rate_threshold,
    )

    final_users = (
        data_filtered["session_uuid"].nunique()
        if "session_uuid" in data_filtered.columns
        else 0
    )
    final_trials = (
        data_filtered[["session_uuid", "trial_id"]].drop_duplicates().shape[0]
        if "session_uuid" in data_filtered.columns
        and "trial_id" in data_filtered.columns
        else 0
    )
    final_user_ids = (
        set(data_filtered["session_uuid"])
        if "session_uuid" in data_filtered.columns
        else set()
    )

    trials_removed = initial_trials - final_trials
    users_removed = initial_users - final_users
    removed_user_ids = sorted(list(initial_user_ids - final_user_ids))

    # Calculate complete users (18 trials)
    trials_per_user = (
        data_filtered.groupby("session_uuid")["trial_id"].nunique()
        if "session_uuid" in data_filtered.columns
        and "trial_id" in data_filtered.columns
        else pd.Series(dtype=int)
    )
    complete_users = int((trials_per_user == 18).sum())

    bad_users = bad_trials.attrs.get("bad_users", [])
    criteria_counts = bad_trials.attrs.get("criteria_counts", {})

    if output_path is None:
        stem = input_path.stem
        output_name = f"{stem}_filtered.tsv"
        out_file = input_path.parent / output_name
    else:
        out_file = Path(output_path).expanduser().resolve()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    data_filtered.to_csv(out_file, sep="\t", index=False)

    print("=" * 70)
    print("DATA FILTERING SUMMARY")
    print("=" * 70)
    print(f"Filtered data saved to: {out_file}\n")
    print(f"Users removed: {users_removed} (of {initial_users} initial participants)")
    if bad_users:
        print(
            f"  - Excluded for rating reference > {ref_score_threshold:.0f} in > {ref_rate_threshold * 100:.0f}% of trials ({len(bad_users)} users):"
        )
        for u in bad_users:
            pct = u["pct_gt_threshold"] * 100
            print(
                f"      * {u['session_uuid']}: {u['n_gt_threshold']}/{u['n_ref_trials']} trials ({pct:.1f}%)"
            )
    other_removed_users = [
        uid
        for uid in removed_user_ids
        if uid not in {u["session_uuid"] for u in bad_users}
    ]
    if other_removed_users:
        print(f"  - Excluded for other reasons ({len(other_removed_users)} users):")
        for uid in other_removed_users:
            print(f"      * {uid}")

    print(f"\nTrials removed: {trials_removed}")
    n_training_trials = (
        df_raw[df_raw.get("trial_id") == "training"][["session_uuid", "trial_id"]]
        .drop_duplicates()
        .shape[0]
        if "trial_id" in df_raw.columns
        else 0
    )
    if n_training_trials > 0:
        print(f"  - Training trials excluded:                     {n_training_trials}")
    n_bad_user_trials = sum(u.get("n_ref_trials", 0) for u in bad_users)
    if n_bad_user_trials > 0:
        print(f"  - Trials from excluded users:                   {n_bad_user_trials}")
    print(f"  - Bad quality trials excluded (retained users): {len(bad_trials)}")
    if criteria_counts:
        print("    Breakdown by criterion (trials may match multiple):")
        for criterion, count in criteria_counts.items():
            print(f"      * {criterion:<22}: {count} trials")
    print(f"\nFinal dataset:")
    print(
        f"  - Initial dataset:  {initial_trials} trials across {initial_users} users ({len(df_raw)} rows)"
    )
    print(
        f"  - Filtered dataset: {final_trials} trials across {final_users} users ({len(data_filtered)} rows)"
    )
    pct_complete = (complete_users / final_users * 100) if final_users > 0 else 0.0
    print(
        f"  - Users with complete data (18 trials): {complete_users} of {final_users} ({pct_complete:.1f}%)"
    )
    print("=" * 70)

    return data_filtered, bad_trials, out_file


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    default_data_path = (
        repo_root / "data" / "listening_test_responses_device_filtered.tsv"
    )
    if not default_data_path.exists():
        default_data_path = repo_root / "data" / "listening_test_responses.tsv"

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
        help="Path to save the filtered TSV file (default: <input_stem>_filtered.tsv)",
    )
    parser.add_argument(
        "--ref-score-threshold",
        type=float,
        default=10.0,
        help="Score threshold above which a reference rating is considered bad (default: 10.0)",
    )
    parser.add_argument(
        "--ref-rate-threshold",
        type=float,
        default=0.15,
        help="Rate threshold above which a user is excluded for bad reference ratings (default: 0.15)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        log.error(f"Data file not found at: {args.data_path}")
        parser.print_help()
        sys.exit(1)

    filter_and_save(
        args.data_path,
        output_path=args.output,
        ref_score_threshold=args.ref_score_threshold,
        ref_rate_threshold=args.ref_rate_threshold,
    )


# ======================================================================
# DATA FILTERING SUMMARY
# ======================================================================
#
# Users removed: 23 (of 51 initial participants)
#   - Excluded for rating reference > 10 in > 15% of trials (23 users):
#       * 029d0004-b12b-4345-b48b-d032e35dc81c: 7/18 trials (38.9%)
#       * 04607afd-671d-4010-a181-d94ca907f1fd: 5/18 trials (27.8%)
#       * 0f3c45e3-38d0-46dd-9dab-e8388e87a0ed: 4/18 trials (22.2%)
#       * 0f4851c3-d8a6-43ba-bfc7-21dde6edc2fc: 5/18 trials (27.8%)
#       * 14f6fc68-9a27-40e5-98ab-b2d1e38ab12e: 3/18 trials (16.7%)
#       * 16bcc0c4-b9ac-4a57-9517-f12fe835be29: 8/18 trials (44.4%)
#       * 18ccb701-2df0-42ce-8980-437612e0bb82: 3/18 trials (16.7%)
#       * 204caf3b-55cb-47a9-920b-cc96988bcaa4: 18/18 trials (100.0%)
#       * 4ce9d4e2-0786-4f01-8a82-dfd08587ed30: 5/18 trials (27.8%)
#       * 4ea66deb-59b1-4e1e-8988-169b5ea04e23: 5/18 trials (27.8%)
#       * 6090cde1-199f-446f-8be0-5af430a526c5: 13/18 trials (72.2%)
#       * 61a2b24f-57b8-43dc-b6d1-d27febadeaaa: 4/18 trials (22.2%)
#       * 84709dd5-605c-4440-85f6-00fc36924a1c: 3/18 trials (16.7%)
#       * 8ab8c143-54ca-4b1a-8035-64c8f1ca9c3a: 8/18 trials (44.4%)
#       * 92d30785-45fe-49ae-8ba1-2d0be1045a4b: 12/18 trials (66.7%)
#       * a75817c8-b3ec-47e0-a505-cf1c9a2101c3: 7/18 trials (38.9%)
#       * aad8d2a2-407f-46fe-91d6-6cf0bca30b7f: 6/18 trials (33.3%)
#       * acf1bde0-2873-4d0b-b1dc-c0896653720c: 3/18 trials (16.7%)
#       * bfb9dd35-eb51-4635-9cbd-ddf058f2d183: 18/18 trials (100.0%)
#       * c7deccbf-b868-4f08-a9ec-6dbfc8c7f3e2: 4/18 trials (22.2%)
#       * cf044467-0a55-49a3-8b71-4e85eced0577: 7/18 trials (38.9%)
#       * d6cf2853-febf-4e8f-89f4-2e598f9637af: 17/18 trials (94.4%)
#       * ff7d76b3-8310-47a8-aa3c-5ebcea2d1cf9: 8/18 trials (44.4%)
#
# Trials removed: 476
#   - Training trials excluded:                     51
#   - Trials from excluded users:                   414
#   - Bad quality trials excluded (retained users): 11
#     Breakdown by criterion (trials may match multiple):
#       * total_time < 24000 ms : 7 trials
#       * rating_range < 10     : 4 trials
#       * all_identical         : 2 trials
#
# Final dataset:
#   - Initial dataset:  969 trials across 51 users (4845 rows)
#   - Filtered dataset: 493 trials across 28 users (2465 rows)
#   - Users with complete data (18 trials): 21 of 28 (75.0%)
# ======================================================================

# Also removed 3 laptop speaker users
