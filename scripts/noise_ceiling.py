"""Noise ceiling calculation for MUSHRA listening test responses.

Computes both individual-level and group-level noise ceilings for:
1. Entire file (all 18 trials, 72 stimuli per participant)
2. Per modulation type (frequency, amplitude, regularity; 6 trials, 24 stimuli each)
3. Per modulation x timbre combination (3 modulations x 3 timbres = 9 conditions; 2 trials, 8 stimuli each)

Definitions:
- Individual Lower Bound (Leave-One-Out):
    Mean correlation of each participant's ratings with the mean ratings of all
    other participants: mean(corr(s_i, mean_{j!=i}(s_j))).
- Individual Upper Bound (Grand Mean):
    Mean correlation of each participant's ratings with the grand mean across all
    participants: mean(corr(s_i, mean_all(s))).
- Group-Level Ceiling (Split-Half with Spearman-Brown Prophecy):
    Reliability of the group average rating estimated via Monte Carlo split-half
    resampling with the Spearman-Brown formula: R = 2 * r_half / (1 + r_half).
    Provided for Pearson (linear), Spearman (rank/monotonic), and Kendall (concordance) metrics.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def prepare_data(
    data_source: Union[str, Path, pd.DataFrame],
    exclude_reference: bool = True,
) -> pd.DataFrame:
    """Load and prepare MUSHRA TSV/CSV data for noise ceiling analyses.

    Extracts `modulation`, `timbre` (feature), and `source` from `trial_id`
    (e.g., 'freq_brightness_real' -> modulation='freq', timbre='brightness', source='real').
    """
    if isinstance(data_source, (str, Path)):
        file_path = Path(data_source).expanduser().resolve()
        log.info(f"Loading responses from {file_path}")
        sep = "\t" if file_path.suffix == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep)
    else:
        df = data_source.copy()

    # Verify required columns
    required = ["session_uuid", "trial_id", "rating_stimulus", "rating_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Exclude reference anchor if requested (typically rated 0 in difference MUSHRA)
    if exclude_reference:
        df = df[df["rating_stimulus"] != "reference"].copy()

    # Drop training trials if present
    df = df[df["trial_id"] != "training"].copy()

    # Parse trial_id components: modulation, timbre, source
    split_cols = df["trial_id"].str.split("_", expand=True)
    if split_cols.shape[1] >= 2:
        df["modulation"] = split_cols[0]
        df["timbre"] = split_cols[1]
    if split_cols.shape[1] >= 3:
        df["source"] = split_cols[2]

    # Combine modulation and timbre (e.g., 'freq_brightness')
    df["mod_timbre"] = df["modulation"] + "_" + df["timbre"]

    return df


def _compute_slice_ceilings(
    matrix: pd.DataFrame,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute individual lower/upper bounds and group split-half ceiling for a stimulus x subject matrix."""
    n_stimuli, n_subjects = matrix.shape
    if n_subjects < 3 or n_stimuli < 3:
        return {
            "n_stimuli": n_stimuli,
            "n_subjects": n_subjects,
            "pearson_lower": float("nan"),
            "pearson_lower_std": float("nan"),
            "pearson_upper": float("nan"),
            "pearson_upper_std": float("nan"),
            "pearson_group": float("nan"),
            "pearson_group_std": float("nan"),
            "spearman_lower": float("nan"),
            "spearman_lower_std": float("nan"),
            "spearman_upper": float("nan"),
            "spearman_upper_std": float("nan"),
            "spearman_group": float("nan"),
            "spearman_group_std": float("nan"),
            "kendall_lower": float("nan"),
            "kendall_lower_std": float("nan"),
            "kendall_upper": float("nan"),
            "kendall_upper_std": float("nan"),
            "kendall_group": float("nan"),
            "kendall_group_std": float("nan"),
        }

    # 1. Individual ceiling (Leave-One-Out vs Grand Mean)
    grand_mean = matrix.mean(axis=1)

    p_lower, p_upper = [], []
    s_lower, s_upper = [], []
    k_lower, k_upper = [], []

    for col in matrix.columns:
        sub = matrix[col]
        loo = matrix.drop(columns=[col]).mean(axis=1)

        # Pearson
        pr_low, _ = stats.pearsonr(sub, loo)
        pr_up, _ = stats.pearsonr(sub, grand_mean)
        p_lower.append(pr_low)
        p_upper.append(pr_up)

        # Spearman
        sr_low, _ = stats.spearmanr(sub, loo)
        sr_up, _ = stats.spearmanr(sub, grand_mean)
        s_lower.append(sr_low)
        s_upper.append(sr_up)

        # Kendall tau
        kr_low, _ = stats.kendalltau(sub, loo)
        kr_up, _ = stats.kendalltau(sub, grand_mean)
        k_lower.append(kr_low)
        k_upper.append(kr_up)

    # 2. Group-level split-half reliability with Spearman-Brown prophecy formula
    rng = np.random.default_rng(seed)
    cols = matrix.columns.to_numpy()
    half = n_subjects // 2

    p_splits = []
    s_splits = []
    k_splits = []

    for _ in range(n_bootstraps):
        shuffled = rng.permutation(cols)
        m1 = matrix[shuffled[:half]].mean(axis=1)
        m2 = matrix[shuffled[half:]].mean(axis=1)

        # Pearson split-half
        pr, _ = stats.pearsonr(m1, m2)
        if not np.isnan(pr) and (1 + pr) != 0:
            p_sb = (2 * pr) / (1 + pr)
            p_splits.append(p_sb)

        # Spearman split-half
        sr, _ = stats.spearmanr(m1, m2)
        if not np.isnan(sr) and (1 + sr) != 0:
            s_sb = (2 * sr) / (1 + sr)
            s_splits.append(s_sb)

        # Kendall split-half
        kr, _ = stats.kendalltau(m1, m2)
        if not np.isnan(kr) and (1 + kr) != 0:
            k_sb = (2 * kr) / (1 + kr)
            k_splits.append(k_sb)

    return {
        "n_stimuli": n_stimuli,
        "n_subjects": n_subjects,
        "pearson_lower": float(np.mean(p_lower)),
        "pearson_lower_std": float(np.std(p_lower)),
        "pearson_upper": float(np.mean(p_upper)),
        "pearson_upper_std": float(np.std(p_upper)),
        "pearson_group": float(np.mean(p_splits)) if p_splits else float("nan"),
        "pearson_group_std": float(np.std(p_splits)) if p_splits else float("nan"),
        "spearman_lower": float(np.mean(s_lower)),
        "spearman_lower_std": float(np.std(s_lower)),
        "spearman_upper": float(np.mean(s_upper)),
        "spearman_upper_std": float(np.std(s_upper)),
        "spearman_group": float(np.mean(s_splits)) if s_splits else float("nan"),
        "spearman_group_std": float(np.std(s_splits)) if s_splits else float("nan"),
        "kendall_lower": float(np.mean(k_lower)),
        "kendall_lower_std": float(np.std(k_lower)),
        "kendall_upper": float(np.mean(k_upper)),
        "kendall_upper_std": float(np.std(k_upper)),
        "kendall_group": float(np.mean(k_splits)) if k_splits else float("nan"),
        "kendall_group_std": float(np.std(k_splits)) if k_splits else float("nan"),
    }


def compute_noise_ceiling(
    data_source: Union[str, Path, pd.DataFrame],
    level: Literal["all", "entire", "modulation", "modulation_timbre"] = "all",
    complete_subjects: Literal["slice", "global"] = "slice",
    exclude_reference: bool = True,
    n_bootstraps: int = 1000,
    seed: int = 42,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute noise ceilings at the specified level(s) of granularity.

    Args:
        data_source: TSV/CSV filepath or prepared DataFrame.
        level: Granularity level:
            - 'entire': 1 overall ceiling across all 18 trials (72 stimuli).
            - 'modulation': 3 ceilings for frequency, amplitude, regularity (6 trials / 24 stimuli each).
            - 'modulation_timbre': 9 ceilings for mod x timbre (2 trials / 8 stimuli each).
            - 'all': computes all 3 levels.
        complete_subjects:
            - 'slice': include subjects with complete data for that specific slice (maximizes N per slice).
            - 'global': include only subjects who completed all 18 trials across the entire file.
        exclude_reference: Whether to drop the reference condition (default: True).
        n_bootstraps: Number of Monte Carlo iterations for group split-half estimation.
        seed: Random seed for split-half reproducibility.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        pd.DataFrame containing summary statistics for each condition slice.
    """
    df = prepare_data(data_source, exclude_reference=exclude_reference)

    # If global complete is requested, find subjects complete across all 18 trials
    if complete_subjects == "global":
        full_pivot = df.pivot_table(
            index=["trial_id", "rating_stimulus"],
            columns="session_uuid",
            values="rating_score",
        )
        valid_subjects = set(full_pivot.dropna(axis=1).columns)
        df = df[df["session_uuid"].isin(valid_subjects)].copy()
        log.info(
            f"Using {len(valid_subjects)} global complete subjects across all analyses."
        )

    tasks: list[tuple[str, str, int, pd.DataFrame]] = []

    # 1. Entire dataset (18 trials)
    if level in ("all", "entire"):
        pivot_entire = df.pivot_table(
            index=["trial_id", "rating_stimulus"],
            columns="session_uuid",
            values="rating_score",
        ).dropna(axis=1)
        tasks.append(("entire", "all_18_trials", 18, pivot_entire))

    # 2. Per modulation type (6 trials each)
    if level in ("all", "modulation"):
        modulations = sorted(df["modulation"].dropna().unique())
        for mod in modulations:
            sub_df = df[df["modulation"] == mod]
            pivot_mod = sub_df.pivot_table(
                index=["trial_id", "rating_stimulus"],
                columns="session_uuid",
                values="rating_score",
            ).dropna(axis=1)
            tasks.append(("modulation", mod, sub_df["trial_id"].nunique(), pivot_mod))

    # 3. Per modulation x timbre (2 trials each)
    if level in ("all", "modulation_timbre"):
        mod_timbres = sorted(df["mod_timbre"].dropna().unique())
        for mt in mod_timbres:
            sub_df = df[df["mod_timbre"] == mt]
            pivot_mt = sub_df.pivot_table(
                index=["trial_id", "rating_stimulus"],
                columns="session_uuid",
                values="rating_score",
            ).dropna(axis=1)
            tasks.append(
                (
                    "modulation_timbre",
                    mt,
                    sub_df["trial_id"].nunique(),
                    pivot_mt,
                )
            )

    records = []
    pbar = tqdm(
        tasks,
        desc="Computing noise ceilings",
        unit="slice",
        disable=not show_progress,
    )
    for granularity, condition, n_trials, pivot_slice in pbar:
        pbar.set_postfix_str(condition)
        res = _compute_slice_ceilings(pivot_slice, n_bootstraps=n_bootstraps, seed=seed)
        res["granularity"] = granularity
        res["condition"] = condition
        res["n_trials"] = n_trials
        records.append(res)

    result_df = pd.DataFrame(records)

    # Reorder columns for readability
    first_cols = [
        "granularity",
        "condition",
        "n_trials",
        "n_stimuli",
        "n_subjects",
    ]
    other_cols = [c for c in result_df.columns if c not in first_cols]
    return result_df[first_cols + other_cols]


if __name__ == "__main__":
    default_data_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "listening_test_responses_preprocessed.tsv"
    )

    parser = argparse.ArgumentParser(
        description="Compute individual and group-level noise ceilings on MUSHRA responses."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=str(default_data_path),
        help=f"Path to input MUSHRA TSV file (default: {default_data_path})",
    )
    parser.add_argument(
        "--level",
        choices=["all", "entire", "modulation", "modulation_timbre"],
        default="all",
        help="Granularity level to compute (default: all)",
    )
    parser.add_argument(
        "--complete-subjects",
        choices=["slice", "global"],
        default="slice",
        help="Subjects to include: 'slice' (complete for that condition) or 'global' (complete for all 18 trials).",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Include reference stimulus rating in the calculation (default: excluded).",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=1000,
        help="Number of Monte Carlo iterations for group split-half estimation (default: 1000).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional path to save results as CSV or TSV.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        log.error(f"File not found: {args.data_path}")
        sys.exit(1)

    results_df = compute_noise_ceiling(
        data_source=args.data_path,
        level=args.level,
        complete_subjects=args.complete_subjects,
        exclude_reference=not args.include_reference,
        n_bootstraps=args.bootstraps,
        show_progress=not args.no_progress,
    )

    # Configure pandas display and print the collected DataFrame directly
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.precision", 3)

    print("\n" + "=" * 120)
    print("NOISE CEILING RESULTS (DataFrame)")
    print("=" * 120)
    print(results_df.to_string(index=False))
    print("=" * 120 + "\n")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sep = "\t" if out_path.suffix == ".tsv" else ","
        results_df.to_csv(out_path, sep=sep, index=False)
        print(f"Results successfully exported to: {out_path}")
