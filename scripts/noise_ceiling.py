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
- Group-Level Ceiling (Monte Carlo Split-Half):
    Reliability of the group average rating estimated via Monte Carlo split-half
    resampling. Spearman-Brown prophecy correction (R = 2 * r / (1 + r)) is applied
    to Pearson and Spearman; raw split-half correlation is reported for Kendall tau.
    All standard deviations are sample standard deviations (ddof=1).
    95% bootstrap confidence intervals are computed via the 2.5th and 97.5th percentiles.
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
            "pearson_group_ci95_low": float("nan"),
            "pearson_group_ci95_high": float("nan"),
            "spearman_lower": float("nan"),
            "spearman_lower_std": float("nan"),
            "spearman_upper": float("nan"),
            "spearman_upper_std": float("nan"),
            "spearman_group": float("nan"),
            "spearman_group_std": float("nan"),
            "spearman_group_ci95_low": float("nan"),
            "spearman_group_ci95_high": float("nan"),
            "kendall_lower": float("nan"),
            "kendall_lower_std": float("nan"),
            "kendall_upper": float("nan"),
            "kendall_upper_std": float("nan"),
            "kendall_group": float("nan"),
            "kendall_group_std": float("nan"),
            "kendall_group_ci95_low": float("nan"),
            "kendall_group_ci95_high": float("nan"),
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

    # 2. Group-level split-half reliability
    # Spearman-Brown prophecy formula is applied to Pearson and Spearman;
    # Kendall tau is left uncorrected (raw split-half concordance).
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

        # Pearson split-half (with Spearman-Brown correction)
        pr, _ = stats.pearsonr(m1, m2)
        if not np.isnan(pr) and (1 + pr) != 0:
            p_sb = (2 * pr) / (1 + pr)
            p_splits.append(p_sb)

        # Spearman split-half (with Spearman-Brown correction)
        sr, _ = stats.spearmanr(m1, m2)
        if not np.isnan(sr) and (1 + sr) != 0:
            s_sb = (2 * sr) / (1 + sr)
            s_splits.append(s_sb)

        # Kendall split-half (raw split-half without Spearman-Brown correction)
        kr, _ = stats.kendalltau(m1, m2)
        if not np.isnan(kr):
            k_splits.append(kr)

    # Compute 95% bootstrap confidence intervals (2.5th and 97.5th percentiles)
    p_ci_low = float(np.percentile(p_splits, 2.5)) if p_splits else float("nan")
    p_ci_high = float(np.percentile(p_splits, 97.5)) if p_splits else float("nan")

    s_ci_low = float(np.percentile(s_splits, 2.5)) if s_splits else float("nan")
    s_ci_high = float(np.percentile(s_splits, 97.5)) if s_splits else float("nan")

    k_ci_low = float(np.percentile(k_splits, 2.5)) if k_splits else float("nan")
    k_ci_high = float(np.percentile(k_splits, 97.5)) if k_splits else float("nan")

    return {
        "n_stimuli": n_stimuli,
        "n_subjects": n_subjects,
        "pearson_lower": float(np.mean(p_lower)),
        "pearson_lower_std": float(np.std(p_lower, ddof=1)),
        "pearson_upper": float(np.mean(p_upper)),
        "pearson_upper_std": float(np.std(p_upper, ddof=1)),
        "pearson_group": float(np.mean(p_splits)) if p_splits else float("nan"),
        "pearson_group_std": float(np.std(p_splits, ddof=1)) if p_splits else float("nan"),
        "pearson_group_ci95_low": p_ci_low,
        "pearson_group_ci95_high": p_ci_high,
        "spearman_lower": float(np.mean(s_lower)),
        "spearman_lower_std": float(np.std(s_lower, ddof=1)),
        "spearman_upper": float(np.mean(s_upper)),
        "spearman_upper_std": float(np.std(s_upper, ddof=1)),
        "spearman_group": float(np.mean(s_splits)) if s_splits else float("nan"),
        "spearman_group_std": float(np.std(s_splits, ddof=1)) if s_splits else float("nan"),
        "spearman_group_ci95_low": s_ci_low,
        "spearman_group_ci95_high": s_ci_high,
        "kendall_lower": float(np.mean(k_lower)),
        "kendall_lower_std": float(np.std(k_lower, ddof=1)),
        "kendall_upper": float(np.mean(k_upper)),
        "kendall_upper_std": float(np.std(k_upper, ddof=1)),
        "kendall_group": float(np.mean(k_splits)) if k_splits else float("nan"),
        "kendall_group_std": float(np.std(k_splits, ddof=1)) if k_splits else float("nan"),
        "kendall_group_ci95_low": k_ci_low,
        "kendall_group_ci95_high": k_ci_high,
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
        res = _compute_slice_ceilings(
            pivot_slice, n_bootstraps=n_bootstraps, seed=seed
        )
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


# ========================================================================================================================
# NOISE CEILING RESULTS (DataFrame)
# ========================================================================================================================
#       granularity       condition  n_trials  n_stimuli  n_subjects  pearson_lower  pearson_lower_std  pearson_upper  pearson_upper_std  pearson_group  pearson_group_std  pearson_group_ci95_low  pearson_group_ci95_high  spearman_lower  spearman_lower_std  spearman_upper  spearman_upper_std  spearman_group  spearman_group_std  spearman_group_ci95_low  spearman_group_ci95_high  kendall_lower  kendall_lower_std  kendall_upper  kendall_upper_std  kendall_group  kendall_group_std  kendall_group_ci95_low  kendall_group_ci95_high
#            entire   all_18_trials        18         72          22          0.820              0.061          0.837              0.056          0.980              0.005                   0.969                    0.987           0.817               0.058           0.833               0.054           0.978               0.006                    0.964                     0.986          0.646              0.062          0.665              0.059          0.832              0.021                   0.783                    0.868
#        modulation             amp         6         24          38          0.784              0.096          0.796              0.092          0.985              0.006                   0.970                    0.993           0.786               0.093           0.790               0.091           0.985               0.007                    0.969                     0.994          0.622              0.096          0.629              0.095          0.874              0.034                   0.804                    0.935
#        modulation            freq         6         24          39          0.876              0.085          0.882              0.081          0.993              0.003                   0.987                    0.997           0.871               0.079           0.878               0.075           0.988               0.004                    0.980                     0.995          0.722              0.098          0.732              0.095          0.896              0.025                   0.848                    0.942
#        modulation             reg         6         24          29          0.793              0.082          0.808              0.076          0.980              0.007                   0.964                    0.990           0.799               0.079           0.811               0.079           0.977               0.008                    0.959                     0.989          0.638              0.081          0.651              0.081          0.838              0.033                   0.768                    0.899
# modulation_timbre  amp_brightness         2          8          44          0.808              0.160          0.816              0.155          0.990              0.007                   0.971                    0.998           0.767               0.195           0.800               0.178           0.973               0.008                    0.963                     0.988          0.661              0.202          0.691              0.192          0.842              0.047                   0.786                    0.929
# modulation_timbre    amp_richness         2          8          45          0.819              0.159          0.827              0.153          0.990              0.007                   0.973                    0.998           0.782               0.187           0.803               0.177           0.968               0.011                    0.950                     0.988          0.661              0.185          0.683              0.180          0.812              0.064                   0.714                    0.929
# modulation_timbre      amp_warmth         2          8          43          0.790              0.160          0.799              0.155          0.987              0.008                   0.967                    0.998           0.791               0.180           0.791               0.180           0.994               0.010                    0.963                     1.000          0.674              0.186          0.674              0.186          0.969              0.047                   0.857                    1.000
# modulation_timbre freq_brightness         2          8          45          0.944              0.050          0.947              0.048          0.998              0.001                   0.995                    1.000           0.930               0.050           0.930               0.050           0.987               0.011                    0.963                     1.000          0.833              0.087          0.833              0.087          0.925              0.063                   0.786                    1.000
# modulation_timbre   freq_richness         2          8          43          0.874              0.161          0.879              0.154          0.993              0.004                   0.985                    0.999           0.870               0.145           0.870               0.145           0.974               0.012                    0.950                     1.000          0.772              0.162          0.773              0.161          0.850              0.070                   0.714                    1.000
# modulation_timbre     freq_warmth         2          8          45          0.885              0.132          0.889              0.127          0.995              0.003                   0.987                    0.999           0.883               0.147           0.883               0.147           0.999               0.003                    0.988                     1.000          0.786              0.173          0.786              0.173          0.995              0.019                   0.929                    1.000
# modulation_timbre  reg_brightness         2          8          39          0.832              0.130          0.841              0.124          0.989              0.006                   0.976                    0.998           0.835               0.143           0.835               0.143           0.984               0.010                    0.963                     1.000          0.723              0.167          0.723              0.167          0.904              0.056                   0.786                    1.000
# modulation_timbre    reg_richness         2          8          42          0.834              0.135          0.842              0.131          0.991              0.005                   0.979                    0.998           0.795               0.184           0.797               0.182           0.976               0.011                    0.950                     1.000          0.678              0.180          0.680              0.178          0.859              0.064                   0.714                    1.000
# modulation_timbre      reg_warmth         2          8          33          0.786              0.193          0.799              0.187          0.982              0.011                   0.952                    0.996           0.797               0.190           0.797               0.190           0.989               0.014                    0.950                     1.000          0.684              0.196          0.684              0.196          0.942              0.063                   0.786                    1.000
# ========================================================================================================================
