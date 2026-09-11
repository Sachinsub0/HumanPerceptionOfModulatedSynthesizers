import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

# Add repo code directory to path
code_dir = Path(__file__).resolve().parent.parent / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from util import parse_amount

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

SERIES_COLOR = "#2a78d6"
AXIS_COLOR = "#52514e"

MOD_SIG_XLABELS = {
    "amp": "Modulation depth",
    "freq": "Modulation rate (Hz)",
    "reg": "Modulation irregularity",
}
# Mod rates are spaced in octaves, the other amounts are spaced linearly
MOD_SIG_LOG_X = {"freq"}
# Larger wavetable groups are named "all" instead of by their common prefix
MAX_NAMED_GROUP_SIZE = 3
FIG_SIZE = (6, 6)
DPI = 150

DEFAULT_WAVETABLES = [
    # "brightness_real__harmonics__synced_sines__256_1024",
    # "brightness_synthetic__256_1024",
    # "richness_real__filter__acid_saw__46_1024__inverted",
    # "richness_synthetic__256_1024",
    # "warmth_real__vintage__logue_saw__166_1024",
    # "warmth_synthetic__256_1024",
    # [
    #     "brightness_real__harmonics__synced_sines__256_1024",
    #     "brightness_synthetic__256_1024",
    # ],
    # [
    #     "richness_real__filter__acid_saw__46_1024__inverted",
    #     "richness_synthetic__256_1024",
    # ],
    # [
    #     "warmth_real__vintage__logue_saw__166_1024",
    #     "warmth_synthetic__256_1024",
    # ],
    [
        "brightness_real__harmonics__synced_sines__256_1024",
        "brightness_synthetic__256_1024",
        "richness_real__filter__acid_saw__46_1024__inverted",
        "richness_synthetic__256_1024",
        "warmth_real__vintage__logue_saw__166_1024",
        "warmth_synthetic__256_1024",
    ],
]


def resolve_group(entry: Union[str, Sequence[str]]) -> Tuple[str, List[str]]:
    """Normalize a wavetables entry into a (group name, wavetable names) pair. A
    list of wavetables is averaged into a single curve and is named after the
    common prefix of its members, e.g. ["brightness_real__...",
    "brightness_synthetic__..."] -> "brightness". Groups of more than
    MAX_NAMED_GROUP_SIZE wavetables are named "all"."""
    if isinstance(entry, str):
        return entry, [entry]
    assert len(entry) > 0, "A wavetable group cannot be empty"
    if len(entry) == 1:
        return entry[0], list(entry)
    if len(entry) > MAX_NAMED_GROUP_SIZE:
        return "all", list(entry)
    group_name = os.path.commonprefix(entry).rstrip("_")
    if not group_name:
        group_name = "__and__".join(entry)
    return group_name, list(entry)


def summarize_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the distances of a group into a mean, standard deviation, and
    min-max range per modulation amount."""
    curve = (
        df.groupby("amount")["distance"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    curve["std"] = curve["std"].fillna(0.0)
    return curve.sort_values("amount")


def compute_ylim(curves: List[pd.DataFrame], pad: float = 0.05) -> Tuple[float, float]:
    """Y range covering every curve of a loss function, including its std
    ranges, so that all of its plots share a comparable axis."""
    lo = min(min((c["mean"] - c["std"]).min(), c["min"].min()) for c in curves)
    hi = max(max((c["mean"] + c["std"]).max(), c["max"].max()) for c in curves)
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def plot_distance_curve(
    curve: pd.DataFrame,
    loss_name: str,
    group_name: str,
    mod_sig: str,
    ylim: Optional[Tuple[float, float]] = None,
    max_shift: int = 0,
    save_dir: str = "",
) -> None:
    mod_type = mod_sig.split("_", 1)[0]
    _, ref_amount, _ = parse_amount(mod_sig)
    n_wt = int(curve["count"].max())

    yerr = None
    if n_wt > 1:
        # Symmetric bars spanning +/- 1 standard deviation of the group
        yerr = curve["std"]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_box_aspect(1)  # Square plotting area, not a square figure
    ax.errorbar(
        curve["amount"],
        curve["mean"],
        yerr=yerr,
        color=SERIES_COLOR,
        linewidth=2.0,
        marker="o",
        markersize=8,
        capsize=4,
        elinewidth=1.5,
    )
    ax.axvline(
        ref_amount,
        color=AXIS_COLOR,
        linewidth=1.0,
        linestyle="--",
        alpha=0.5,
        label=f"reference = {ref_amount:g}",
    )
    ax.legend(loc="best", frameon=False, fontsize=8, labelcolor=AXIS_COLOR)
    if mod_type in MOD_SIG_LOG_X:
        ax.set_xscale("log", base=2)
        ticks = sorted(set(curve["amount"].tolist() + [ref_amount]))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(MOD_SIG_XLABELS[mod_type])
    ax.set_ylabel(f"{loss_name} distance")
    title = f"{loss_name} distance from {mod_sig}\n{group_name}"
    if n_wt > 1:
        title += f"\nmean of {n_wt} wavetables with ±1 SD range"
    else:
        # Kept 3 lines tall so every plot ends up the same size
        title += "\nsingle wavetable"
    if max_shift > 0:
        title += f"\nref phase-shifted by 0-{max_shift} samples"
    ax.set_title(title, fontsize=10)
    ax.grid(True, color=AXIS_COLOR, alpha=0.15, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    # plt.show()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_name = f"{loss_name}__{group_name}__{mod_sig}.png"
        plt.savefig(os.path.join(save_dir, save_name), dpi=DPI)
        log.info(f"Saved {save_name}")
    plt.close(fig)


def plot_distance_groups(
    tsv_path: Union[str, pd.DataFrame],
    groups: Sequence[Tuple[str, List[str]]],
    save_dir: str,
    max_shift: int = 0,
) -> None:
    """Read distances from TSV/CSV or DataFrame, aggregate into the defined groups,
    and plot curves."""
    if isinstance(tsv_path, (str, Path)):
        file_path = str(tsv_path)
        sep = "\t" if file_path.endswith(".tsv") else ","
        df = pd.read_csv(file_path, sep=sep)
        log.info(f"Loaded {len(df)} distances from {file_path} for plotting groups")
    else:
        df = tsv_path

    os.makedirs(save_dir, exist_ok=True)

    n_plots = 0
    for loss_name, loss_df in df.groupby("loss_fn", sort=False):
        curves = {}
        for group_name, wt_names in groups:
            group_df = loss_df[loss_df["wavetable"].isin(wt_names)]
            if group_df.empty:
                log.warning(
                    f"No data found for group '{group_name}' with wavetables {wt_names}"
                )
                continue
            for ref_mod_sig, ref_df in group_df.groupby("reference", sort=False):
                curves[(group_name, ref_mod_sig)] = summarize_curve(ref_df)

        if not curves:
            continue

        ylim = compute_ylim(list(curves.values()))
        log.info(f"{loss_name} ylim = ({ylim[0]:.6g}, {ylim[1]:.6g})")
        for (group_name, mod_sig), curve in curves.items():
            plot_distance_curve(
                curve,
                loss_name,
                group_name,
                mod_sig,
                ylim=ylim,
                max_shift=max_shift,
                save_dir=save_dir,
            )
            n_plots += 1
    log.info(f"Saved {n_plots} plots to {save_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot perceptual loss distance curves from distances TSV/CSV."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="../out/distances/distances.tsv",
        help="Path to distances TSV or CSV (default: ../out/distances/distances.tsv, fallback: data/distances__all.csv)",
    )
    parser.add_argument(
        "--save-dir",
        "-o",
        default="../out/distances",
        help="Directory to save generated plot PNGs (default: ../out/distances)",
    )
    parser.add_argument(
        "--max-shift",
        type=int,
        default=0,
        help="Reported phase shift in title (default: 0)",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        repo_data_fallback = (
            Path(__file__).resolve().parent.parent / "data" / "distances__all.csv"
        )
        if repo_data_fallback.exists():
            log.warning(f"{input_path} not found; falling back to {repo_data_fallback}")
            input_path = str(repo_data_fallback)
        else:
            raise FileNotFoundError(f"Input distances file not found: {input_path}")

    groups = [resolve_group(entry) for entry in DEFAULT_WAVETABLES]
    plot_distance_groups(
        tsv_path=input_path,
        groups=groups,
        save_dir=args.save_dir,
        max_shift=args.max_shift,
    )


if __name__ == "__main__":
    main()
