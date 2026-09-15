"""Unpooled 4-way Repeated-Measures ANOVA on Audio Distance Loss Functions across Phases.

Treats each phase variation in data/distances_phases_23.tsv as an individual participant
response (analogous to `session_uuid` in data/listening_test_responses_postprocessed.tsv).
For each loss function, computes a 4-way unpooled repeated-measures ANOVA:
    modulation x feature x source x rating_stimulus
with:
    - modulation: 3 levels (amp, freq, reg)
    - feature: 3 levels (brightness, richness, warmth)
    - source: 2 levels (real, synthetic)
    - rating_stimulus: 4 levels (condition_a, condition_b, condition_c, condition_d)
    - subject: phase (23 levels: 0 to 22)
    - dependent variable: distance

Effect sizes reported:
    - np2: Partial eta-squared
    - ng2: Generalized eta-squared
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence, Union

import pandas as pd

# Ensure scripts directory is in sys.path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anovas import _filter_balanced_subjects, compute_rm_anova_with_effect_sizes

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# Physical amount mapping to MUSHRA condition levels
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


def _parse_wavetable_feature_source(wavetable: str) -> tuple[str, str]:
    """Extract (feature, source) from wavetable string."""
    prefix = wavetable.split("__")[0]
    parts = prefix.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    raise ValueError(f"Unable to parse feature and source from wavetable: {wavetable}")


def prepare_phase_distances(
    data_source: Union[str, Path, pd.DataFrame],
    exclude_reference: bool = True,
) -> pd.DataFrame:
    """Load and prepare phase-varying audio distances dataset for RM-ANOVA.

    Maps:
    - `wavetable` -> `feature` (brightness, richness, warmth) and `source` (real, synthetic)
    - `mod_type` -> `modulation` (amp, freq, reg)
    - `amount` -> `rating_stimulus` (condition_a, condition_b, condition_c, condition_d)
    Excludes reference stimulus ratings (where amount == ref_amount) if exclude_reference is True.
    """
    if isinstance(data_source, (str, Path)):
        file_path = Path(data_source).expanduser().resolve()
        log.info(f"Loading phase distance dataset from {file_path}")
        sep = "\t" if file_path.suffix == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep)
    else:
        df = data_source.copy()

    required_cols = ["loss_fn", "wavetable", "mod_type", "amount", "phase", "distance"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Exclude reference anchor if requested
    if exclude_reference:
        if "ref_amount" in df.columns:
            df = df[df["amount"] != df["ref_amount"]].copy()
        elif "is_reference" in df.columns:
            df = df[~df["is_reference"]].copy()

    # Extract feature and source if not already present
    if "feature" not in df.columns or "source" not in df.columns:
        features, sources = zip(*df["wavetable"].map(_parse_wavetable_feature_source))
        df["feature"] = features
        df["source"] = sources

    # Extract modulation if not already present
    if "modulation" not in df.columns:
        df["modulation"] = df["mod_type"]

    # Map amount to rating_stimulus if not already present
    if "rating_stimulus" not in df.columns:
        stimulus_list = []
        for _, row in df.iterrows():
            mod = row["modulation"]
            amt = float(row["amount"])
            mod_map = AMOUNT_TO_STIMULUS.get(mod, {})
            matched = None
            for ref_amt, cond_name in mod_map.items():
                if abs(amt - ref_amt) < 1e-4:
                    matched = cond_name
                    break
            if matched is None:
                raise ValueError(f"Unmapped amount {amt} for modulation '{mod}'")
            stimulus_list.append(matched)
        df["rating_stimulus"] = stimulus_list

    if exclude_reference:
        df = df[df["rating_stimulus"] != "reference"].copy()

    return df


def compute_loss_anovas(
    data_source: Union[str, Path, pd.DataFrame],
    loss_fns: Union[str, Sequence[str]] = "all",
    subject_col: str = "phase",
    dv_col: str = "distance",
    exclude_reference: bool = True,
) -> pd.DataFrame:
    """Compute 4-way unpooled Repeated-Measures ANOVA for each loss function across phases.

    Design:
        within: ["modulation", "feature", "source", "rating_stimulus"]
        subject: phase
        dependent variable: distance

    Returns:
        pd.DataFrame containing ANOVA tables for each loss function with np2 and ng2.
    """
    df = prepare_phase_distances(data_source, exclude_reference=exclude_reference)

    available_losses = sorted(df["loss_fn"].unique())
    if isinstance(loss_fns, str) and loss_fns.lower() == "all":
        selected_losses = available_losses
    elif isinstance(loss_fns, str):
        selected_losses = [l.strip() for l in loss_fns.split(",") if l.strip()]
    else:
        selected_losses = list(loss_fns)

    within = ["modulation", "feature", "source", "rating_stimulus"]

    all_tables: list[pd.DataFrame] = []

    for loss_name in selected_losses:
        log.info(f"Computing 4-way unpooled RM-ANOVA for loss function: {loss_name}...")
        sub_df = df[df["loss_fn"] == loss_name].copy()
        if sub_df.empty:
            log.warning(f"No data found for loss function '{loss_name}', skipping.")
            continue

        # Ensure design is balanced across phases
        sub_df_bal = _filter_balanced_subjects(
            sub_df, subject_col=subject_col, group_cols=within, dv=dv_col
        )

        aov = compute_rm_anova_with_effect_sizes(
            sub_df_bal,
            dv=dv_col,
            subject=subject_col,
            within=within,
        )
        aov["DF1"] = aov["DF1"].astype(int)
        aov["DF2"] = aov["DF2"].astype(int)
        aov.insert(0, "loss_fn", loss_name)
        all_tables.append(aov)

    if not all_tables:
        return pd.DataFrame()

    return pd.concat(all_tables, ignore_index=True)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    default_input_path = repo_root / "data" / "distances_phases_23.tsv"
    default_output_path = repo_root / "data" / "distances_phases_23_anovas.tsv"

    parser = argparse.ArgumentParser(
        description="Compute 4-way unpooled Repeated-Measures ANOVA on audio distances across phases."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(default_input_path),
        help=f"Path to input phase distances TSV/CSV file (default: {default_input_path})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(default_output_path),
        help=f"Path to save output ANOVA results TSV (default: {default_output_path})",
    )
    parser.add_argument(
        "-l",
        "--loss-fn",
        "--loss-fns",
        nargs="+",
        default=["all"],
        help="Specific loss function(s) to evaluate (default: all)",
    )
    args = parser.parse_args()

    input_file = Path(args.input_path).expanduser().resolve()
    if not input_file.exists():
        log.error(f"Input file not found: {input_file}")
        sys.exit(1)

    loss_fn_arg = args.loss_fn
    if len(loss_fn_arg) == 1 and loss_fn_arg[0].lower() == "all":
        loss_fn_arg = "all"

    results_df = compute_loss_anovas(
        data_source=input_file,
        loss_fns=loss_fn_arg,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\n" + "=" * 110)
    print("4-WAY UNPOOLED REPEATED MEASURES ANOVA RESULTS FOR LOSS FUNCTIONS ACROSS PHASES")
    print("=" * 110)
    for loss_name, group in results_df.groupby("loss_fn", sort=False):
        print(f"\n--- Loss Function: {loss_name} ---")
        disp_df = group.drop(columns=["loss_fn"]).copy()
        print(disp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n" + "=" * 110 + "\n")

    if args.output:
        output_file = Path(args.output).expanduser().resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, sep="\t", index=False)
        log.info(f"Successfully saved ANOVA results to: {output_file}")
