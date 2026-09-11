"""Correlation analysis between audio distance functions and human MUSHRA listening test data.

Compares audio distance functions (MSE, MSS, MFCC, CLAP, PANNs, Wavelet Scattering, JTFS, etc.)
from data/distances__all.csv against human perceptual difference ratings from
data/listening_test_responses_preprocessed.tsv.

Granularity Levels:
1. Entire dataset (all 18 trials, 72 non-reference stimuli per participant)
2. Per modulation type (frequency, amplitude, regularity; 6 trials, 24 stimuli each)
3. Per modulation x timbre combination (3 modulations x 3 timbres = 9 conditions; 2 trials, 8 stimuli each)

Metrics:
- Group Correlation:
    Correlation of model distance with the group mean human ratings across participants:
    - Pearson r (linear alignment) and two-sided p-value
    - Spearman rho (monotonic rank alignment) and two-sided p-value
    - Kendall tau (pairwise concordance) and two-sided p-value
- Individual Correlation:
    Correlation of model distance with each participant's individual ratings:
    - Mean and unbiased sample standard deviation (ddof=1) across participants.
- Noise Ceiling Benchmarking (Optional with --include-noise-ceiling):
    Attaches the individual lower/upper bounds and group noise ceilings from noise_ceiling.py
    for direct evaluation against the theoretical human consensus limit.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Literal, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Ensure local imports from scripts directory work cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_ceiling import _compute_slice_ceilings, prepare_data

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# Mapping of modulation type and physical amount to MUSHRA rating_stimulus
AMOUNT_TO_STIMULUS = {
    "amp": {
        0.10: "reference",
        0.30: "condition_a",
        0.50: "condition_b",
        0.70: "condition_c",
        0.90: "condition_d",
    },
    "freq": {
        0.25: "reference",
        0.50: "condition_a",
        1.00: "condition_b",
        2.00: "condition_c",
        4.00: "condition_d",
    },
    "reg": {
        0.000: "reference",
        0.125: "condition_a",
        0.250: "condition_b",
        0.375: "condition_c",
        0.500: "condition_d",
    },
}


def prepare_distances(
    distances_source: Union[str, Path, pd.DataFrame],
    exclude_reference: bool = True,
) -> pd.DataFrame:
    """Load and parse audio distance data for correlation analysis.

    Extracts `trial_id` and `rating_stimulus` to match the MUSHRA response format:
    - `trial_id`: <modulation>_<timbre>_<source> (e.g. 'freq_brightness_real')
    - `rating_stimulus`: 'condition_a', 'condition_b', 'condition_c', 'condition_d', or 'reference'
    """
    if isinstance(distances_source, (str, Path)):
        file_path = Path(distances_source).expanduser().resolve()
        log.info(f"Loading distances from {file_path}")
        sep = "\t" if file_path.suffix == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep)
    else:
        df = distances_source.copy()

    required = ["loss_fn", "wavetable", "mod_type", "amount", "distance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in distances dataset: {missing}")

    records = []
    timbres = ["brightness", "richness", "warmth"]
    sources = ["real", "synthetic"]

    for _, row in df.iterrows():
        wt = str(row["wavetable"])
        mod = str(row["mod_type"])

        # Identify timbre and source from wavetable string prefix
        timbre, source = None, None
        for t in timbres:
            for s in sources:
                if wt.startswith(f"{t}_{s}"):
                    timbre, source = t, s
                    break
            if timbre is not None:
                break

        if timbre is None or source is None:
            raise ValueError(f"Cannot parse timbre/source from wavetable name: {wt}")

        trial_id = f"{mod}_{timbre}_{source}"
        amount = float(row["amount"])
        is_ref = bool(row.get("is_reference", False))

        if is_ref:
            stimulus = "reference"
        else:
            mod_map = AMOUNT_TO_STIMULUS.get(mod, {})
            # Match amount with floating point tolerance
            stimulus = None
            for ref_val, cond_name in mod_map.items():
                if abs(amount - ref_val) < 1e-4:
                    stimulus = cond_name
                    break
            if stimulus is None:
                raise ValueError(
                    f"Unknown amount {amount} for modulation '{mod}' in {wt}"
                )

        records.append(
            {
                "loss_fn": str(row["loss_fn"]),
                "trial_id": trial_id,
                "rating_stimulus": stimulus,
                "modulation": mod,
                "timbre": timbre,
                "source": source,
                "mod_timbre": f"{mod}_{timbre}",
                "amount": amount,
                "is_reference": is_ref or (stimulus == "reference"),
                "distance": float(row["distance"]),
            }
        )

    parsed_df = pd.DataFrame(records)

    if exclude_reference:
        parsed_df = parsed_df[~parsed_df["is_reference"]].copy()

    return parsed_df


def _compute_slice_correlations(
    human_matrix: pd.DataFrame,
    dist_series: pd.Series,
) -> dict[str, float]:
    """Compute group-level and individual-level correlations between distances and human ratings.

    Args:
        human_matrix: Stimulus x subject rating DataFrame (index: ['trial_id', 'rating_stimulus']).
        dist_series: Model distances aligned to human_matrix index.

    Returns:
        dict containing Pearson, Spearman, and Kendall metrics with p-values and individual distributions.
    """
    n_stimuli, n_subjects = human_matrix.shape
    aligned_dist = dist_series.loc[human_matrix.index]

    if n_stimuli < 3 or n_subjects < 1 or aligned_dist.nunique() <= 1:
        return {
            "n_stimuli": n_stimuli,
            "n_subjects": n_subjects,
            "pearson_group": float("nan"),
            "pearson_group_p": float("nan"),
            "spearman_group": float("nan"),
            "spearman_group_p": float("nan"),
            "kendall_group": float("nan"),
            "kendall_group_p": float("nan"),
            "pearson_indiv": float("nan"),
            "pearson_indiv_std": float("nan"),
            "spearman_indiv": float("nan"),
            "spearman_indiv_std": float("nan"),
            "kendall_indiv": float("nan"),
            "kendall_indiv_std": float("nan"),
        }

    # 1. Group-level correlation (loss distance vs grand mean human rating)
    group_mean = human_matrix.mean(axis=1)

    pr_g, p_val_pr = stats.pearsonr(aligned_dist, group_mean)
    sr_g, p_val_sr = stats.spearmanr(aligned_dist, group_mean)
    kr_g, p_val_kr = stats.kendalltau(aligned_dist, group_mean)

    # 2. Individual-level correlation (loss distance vs each individual participant)
    indiv_p = []
    indiv_s = []
    indiv_k = []

    for col in human_matrix.columns:
        sub_ratings = human_matrix[col]
        if sub_ratings.nunique() <= 1:
            continue
        p_sub, _ = stats.pearsonr(aligned_dist, sub_ratings)
        s_sub, _ = stats.spearmanr(aligned_dist, sub_ratings)
        k_sub, _ = stats.kendalltau(aligned_dist, sub_ratings)
        indiv_p.append(p_sub)
        indiv_s.append(s_sub)
        indiv_k.append(k_sub)

    has_indiv = len(indiv_p) >= 2
    return {
        "n_stimuli": n_stimuli,
        "n_subjects": n_subjects,
        "pearson_group": float(pr_g),
        "pearson_group_p": float(p_val_pr),
        "spearman_group": float(sr_g),
        "spearman_group_p": float(p_val_sr),
        "kendall_group": float(kr_g),
        "kendall_group_p": float(p_val_kr),
        "pearson_indiv": float(np.mean(indiv_p)) if indiv_p else float("nan"),
        "pearson_indiv_std": float(np.std(indiv_p, ddof=1)) if has_indiv else float("nan"),
        "spearman_indiv": float(np.mean(indiv_s)) if indiv_s else float("nan"),
        "spearman_indiv_std": float(np.std(indiv_s, ddof=1)) if has_indiv else float("nan"),
        "kendall_indiv": float(np.mean(indiv_k)) if indiv_k else float("nan"),
        "kendall_indiv_std": float(np.std(indiv_k, ddof=1)) if has_indiv else float("nan"),
    }


def compute_correlations(
    distances_source: Union[str, Path, pd.DataFrame],
    mushra_source: Union[str, Path, pd.DataFrame],
    level: Literal["all", "entire", "modulation", "modulation_timbre"] = "all",
    loss_fn: Union[str, Sequence[str]] = "all",
    complete_subjects: Literal["slice", "global"] = "slice",
    exclude_reference: bool = True,
    include_noise_ceiling: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute correlations between audio loss functions and human perceptual ratings.

    Args:
        distances_source: CSV/TSV path or DataFrame of audio loss distances.
        mushra_source: TSV/CSV path or DataFrame of preprocessed MUSHRA responses.
        level: Granularity level ('entire', 'modulation', 'modulation_timbre', or 'all').
        loss_fn: Specific loss function name(s) or 'all' to evaluate all available losses.
        complete_subjects: 'slice' (complete data for that condition) or 'global' (complete across all 18 trials).
        exclude_reference: Exclude reference stimulus rating (default: True).
        include_noise_ceiling: Compute and append noise ceiling benchmark columns from noise_ceiling.py.
        show_progress: Display tqdm progress bar.

    Returns:
        pd.DataFrame containing full correlation results.
    """
    df_dist = prepare_distances(distances_source, exclude_reference=exclude_reference)
    df_human = prepare_data(mushra_source, exclude_reference=exclude_reference)

    # Filter by global complete subjects if requested
    if complete_subjects == "global":
        full_pivot = df_human.pivot_table(
            index=["trial_id", "rating_stimulus"],
            columns="session_uuid",
            values="rating_score",
        )
        valid_subjects = set(full_pivot.dropna(axis=1).columns)
        df_human = df_human[df_human["session_uuid"].isin(valid_subjects)].copy()
        log.info(
            f"Using {len(valid_subjects)} global complete subjects across all analyses."
        )

    # Determine loss functions to evaluate
    all_available_losses = sorted(df_dist["loss_fn"].unique())
    if loss_fn == "all":
        selected_losses = all_available_losses
    elif isinstance(loss_fn, str):
        selected_losses = [loss_fn]
    else:
        selected_losses = list(loss_fn)

    unknown_losses = set(selected_losses) - set(all_available_losses)
    if unknown_losses:
        raise ValueError(
            f"Unknown loss function(s): {unknown_losses}. Available: {all_available_losses}"
        )

    # Prepare condition slice definitions
    slices: list[tuple[str, str, int, pd.DataFrame]] = []

    # 1. Entire dataset (18 trials)
    if level in ("all", "entire"):
        piv_entire = df_human.pivot_table(
            index=["trial_id", "rating_stimulus"],
            columns="session_uuid",
            values="rating_score",
        ).dropna(axis=1)
        slices.append(("entire", "all_18_trials", 18, piv_entire))

    # 2. Per modulation type (6 trials each)
    if level in ("all", "modulation"):
        for mod in sorted(df_human["modulation"].dropna().unique()):
            sub_m = df_human[df_human["modulation"] == mod]
            piv_mod = sub_m.pivot_table(
                index=["trial_id", "rating_stimulus"],
                columns="session_uuid",
                values="rating_score",
            ).dropna(axis=1)
            slices.append(("modulation", mod, sub_m["trial_id"].nunique(), piv_mod))

    # 3. Per modulation x timbre (2 trials each)
    if level in ("all", "modulation_timbre"):
        for mt in sorted(df_human["mod_timbre"].dropna().unique()):
            sub_mt = df_human[df_human["mod_timbre"] == mt]
            piv_mt = sub_mt.pivot_table(
                index=["trial_id", "rating_stimulus"],
                columns="session_uuid",
                values="rating_score",
            ).dropna(axis=1)
            slices.append(
                ("modulation_timbre", mt, sub_mt["trial_id"].nunique(), piv_mt)
            )

    # Cache noise ceiling calculations if requested
    ceilings_cache: dict[tuple[str, str], dict[str, float]] = {}
    if include_noise_ceiling:
        log.info("Precomputing noise ceilings across condition slices...")
        for granularity, cond_name, _, human_matrix in slices:
            ceilings_cache[(granularity, cond_name)] = _compute_slice_ceilings(
                human_matrix, n_bootstraps=1000, seed=42
            )

    records = []
    total_tasks = len(selected_losses) * len(slices)
    pbar = tqdm(
        total=total_tasks,
        desc="Computing loss correlations",
        unit="eval",
        disable=not show_progress,
    )

    for loss_name in selected_losses:
        loss_df = df_dist[df_dist["loss_fn"] == loss_name].set_index(
            ["trial_id", "rating_stimulus"]
        )

        for granularity, condition, n_trials, human_matrix in slices:
            pbar.set_postfix_str(f"{loss_name} | {condition}")

            res = _compute_slice_correlations(human_matrix, loss_df["distance"])
            res["loss_fn"] = loss_name
            res["granularity"] = granularity
            res["condition"] = condition
            res["n_trials"] = n_trials

            if include_noise_ceiling:
                nc = ceilings_cache.get((granularity, condition), {})
                res["pearson_nc_lower"] = nc.get("pearson_lower", float("nan"))
                res["pearson_nc_upper"] = nc.get("pearson_upper", float("nan"))
                res["pearson_nc_group"] = nc.get("pearson_group", float("nan"))
                res["spearman_nc_lower"] = nc.get("spearman_lower", float("nan"))
                res["spearman_nc_upper"] = nc.get("spearman_upper", float("nan"))
                res["spearman_nc_group"] = nc.get("spearman_group", float("nan"))
                res["kendall_nc_lower"] = nc.get("kendall_lower", float("nan"))
                res["kendall_nc_upper"] = nc.get("kendall_upper", float("nan"))
                res["kendall_nc_group"] = nc.get("kendall_group", float("nan"))

            records.append(res)
            pbar.update(1)

    pbar.close()

    result_df = pd.DataFrame(records)

    first_cols = [
        "loss_fn",
        "granularity",
        "condition",
        "n_trials",
        "n_stimuli",
        "n_subjects",
    ]
    other_cols = [c for c in result_df.columns if c not in first_cols]
    return result_df[first_cols + other_cols]


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    default_distances_path = repo_root / "data" / "distances__all.csv"
    default_mushra_path = (
        repo_root / "data" / "listening_test_responses_preprocessed.tsv"
    )

    parser = argparse.ArgumentParser(
        description="Compute correlations between audio distance loss functions and MUSHRA listening test responses."
    )
    parser.add_argument(
        "distances_path",
        nargs="?",
        default=str(default_distances_path),
        help=f"Path to input audio distances CSV (default: {default_distances_path})",
    )
    parser.add_argument(
        "mushra_path",
        nargs="?",
        default=str(default_mushra_path),
        help=f"Path to input MUSHRA TSV file (default: {default_mushra_path})",
    )
    parser.add_argument(
        "--level",
        choices=["all", "entire", "modulation", "modulation_timbre"],
        default="entire",
        help="Granularity level to compute (default: all)",
    )
    parser.add_argument(
        "--loss-fn",
        default="all",
        help="Specific loss function to evaluate (or 'all' for all loss functions, default: all)",
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
        "--include-noise-ceiling",
        action="store_true",
        help="Include human noise ceilings as benchmark columns in the results.",
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

    if not os.path.exists(args.distances_path):
        log.error(f"Distances file not found: {args.distances_path}")
        sys.exit(1)
    if not os.path.exists(args.mushra_path):
        log.error(f"MUSHRA file not found: {args.mushra_path}")
        sys.exit(1)

    results_df = compute_correlations(
        distances_source=args.distances_path,
        mushra_source=args.mushra_path,
        level=args.level,
        loss_fn=args.loss_fn,
        complete_subjects=args.complete_subjects,
        exclude_reference=not args.include_reference,
        include_noise_ceiling=args.include_noise_ceiling,
        show_progress=not args.no_progress,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.precision", 3)

    print("\n" + "=" * 120)
    print("AUDIO DISTANCE CORRELATION WITH HUMAN PERCEPTUAL DATA (DataFrame)")
    print("=" * 120)
    print(results_df.to_string(index=False))
    print("=" * 120 + "\n")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sep = "\t" if out_path.suffix == ".tsv" else ","
        results_df.to_csv(out_path, sep=sep, index=False)
        print(f"Results successfully exported to: {out_path}")
