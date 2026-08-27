import logging
import os
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as tr
import torchaudio
from auraloss.freq import MultiResolutionSTFTLoss
from torch import Tensor as T
from torch import nn

from util import find_variants, parse_amount
from losses import (
    MFCCDistance,
    PANNsEmbeddingLoss,
    ClapEmbeddingLoss,
    LogMSSLoss,
    Scat1DLoss,
    JTFSTLoss,
)

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


def load_audio(path: str, sr: int) -> T:
    audio, audio_sr = torchaudio.load(path)
    assert audio_sr == sr, f"Expected sr={sr}, got {audio_sr} for {path}"
    # The samples are mono duplicated across both channels
    audio = audio[:1, :]
    return audio.unsqueeze(0)


def phase_shift_audio(audio: T, n_samples: int) -> T:
    """Circularly shift audio to simulate a phase shift. The samples are faded in
    and out, so wrapping around introduces almost no discontinuity."""
    return tr.roll(audio, shifts=n_samples, dims=-1)


def resolve_loss_fn(
    entry: Union[nn.Module, Tuple[str, nn.Module]],
) -> Tuple[str, nn.Module]:
    """Normalize a loss_fns entry into a (name, loss function) pair. The name is
    only used for logging and labelling, so a bare loss function falls back to
    its class name."""
    if isinstance(entry, tuple):
        name, loss_fn = entry
        return name, loss_fn
    return entry.__class__.__name__, entry


def resolve_group(entry: Union[str, List[str]]) -> Tuple[str, List[str]]:
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
    """Aggregate the distances of a group into a mean and a min-max range per
    modulation amount."""
    curve = (
        df.groupby("amount")["distance"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
    )
    return curve.sort_values("amount")


def compute_ylim(curves: List[pd.DataFrame], pad: float = 0.05) -> Tuple[float, float]:
    """Y range covering every curve of a loss function, including its min-max
    ranges, so that all of its plots share a comparable axis."""
    lo = min(c["min"].min() for c in curves)
    hi = max(c["max"].max() for c in curves)
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
        # Asymmetric bars spanning the min and max of the group
        yerr = np.stack([curve["mean"] - curve["min"], curve["max"] - curve["mean"]])

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
        title += f"\nmean of {n_wt} wavetables with min-max range"
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
        save_name = f"{loss_name}__{group_name}__{mod_sig}.png"
        plt.savefig(os.path.join(save_dir, save_name), dpi=DPI)
        log.info(f"Saved {save_name}")
    plt.close(fig)


if __name__ == "__main__":
    samples_dir = "../out/samples_all"
    save_dir = "../out/distances"
    sr = 44100
    target_lufs = -18
    use_rand_phase_shift = False
    max_shift = 2048  # Two wavetable frames (44100 / 1024 Hz carrier)
    shift_seed = 42
    loss_fns = [
        ("mse", nn.MSELoss()),
        ("mss", MultiResolutionSTFTLoss()),
        # (
        #     "mss_rev",
        #     LogMSSLoss(
        #         fft_sizes=[67, 127, 257, 509, 1021, 2053],
        #         hop_sizes=[33, 63, 128, 254, 510, 1026],
        #         win_lengths=[67, 127, 257, 509, 1021, 2053],
        #         window="flat_top",
        #         log_mag_eps=1.0,
        #         gamma=1.0,
        #         p=2,
        #     ),
        # ),
        # ("mfcc", MFCCDistance(sr=sr)),
        # ("clap", ClapEmbeddingLoss(use_cuda=False, in_sr=sr)),
        # ("panns_cnn14_32k", PANNsEmbeddingLoss(variant="cnn14-32k", in_sr=sr)),
        # (
        #     "panns_wavegram_logmel",
        #     PANNsEmbeddingLoss(variant="wavegram-logmel", in_sr=sr),
        # ),
        # ("scat1d", Scat1DLoss(shape=176400, J=12, Q1=8, Q2=2, T=None, max_order=2, p=2)),
        # ("jtfs", JTFSTLoss(shape=176400, J=12, Q1=8, Q2=2, J_fr=3, Q_fr=2, T=None, F=None, format_="joint", p=2)),
    ]
    wavetables = [
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
    mod_sig_references = [
        "amp_1.00hz_0.10",
        "freq_0.25hz",
        "reg_1.00hz_0.000",
    ]

    os.makedirs(save_dir, exist_ok=True)
    suffix = f"_{target_lufs}lufs.wav"
    rand_gen = tr.Generator().manual_seed(shift_seed)

    groups = [resolve_group(entry) for entry in wavetables]
    group_names = [name for name, _ in groups]
    assert len(set(group_names)) == len(
        group_names
    ), f"Wavetable group names must be unique, got {group_names}"
    loss_names = [resolve_loss_fn(entry)[0] for entry in loss_fns]
    assert len(set(loss_names)) == len(
        loss_names
    ), f"Loss function names must be unique, got {loss_names}"

    rows = []
    for loss_entry in loss_fns:
        loss_name, loss_fn = resolve_loss_fn(loss_entry)
        for group_name, wt_names in groups:
            for wt_name in wt_names:
                for mod_sig in mod_sig_references:
                    ref_path = os.path.join(
                        samples_dir, f"{wt_name}__{mod_sig}{suffix}"
                    )
                    assert os.path.exists(ref_path), f"Missing reference {ref_path}"
                    ref_audio = load_audio(ref_path, sr)
                    _, ref_amount, _ = parse_amount(mod_sig)

                    variant_paths = find_variants(samples_dir, wt_name, mod_sig, suffix)
                    log.info(
                        f"{loss_name} | {wt_name} | {mod_sig}: "
                        f"found {len(variant_paths)} samples"
                    )
                    for variant_path in variant_paths:
                        variant_name = os.path.basename(variant_path)[: -len(suffix)]
                        variant_mod_sig = variant_name[len(f"{wt_name}__") :]
                        _, amount, _ = parse_amount(variant_mod_sig)
                        audio = load_audio(variant_path, sr)
                        assert (
                            audio.shape == ref_audio.shape
                        ), f"Shape mismatch: {audio.shape} vs {ref_audio.shape}"
                        # Optionally simulate a phase shift by shifting the reference
                        if use_rand_phase_shift:
                            shift = int(
                                tr.randint(
                                    low=0,
                                    high=max_shift + 1,
                                    size=(1,),
                                    generator=rand_gen,
                                ).item()
                            )
                        else:
                            shift = 0
                        with tr.no_grad():
                            dist = loss_fn(
                                audio, phase_shift_audio(ref_audio, shift)
                            ).item()
                        rows.append(
                            {
                                "loss_fn": loss_name,
                                "group": group_name,
                                "wavetable": wt_name,
                                "mod_type": mod_sig.split("_", 1)[0],
                                "reference": mod_sig,
                                "ref_amount": ref_amount,
                                "mod_sig": variant_mod_sig,
                                "amount": amount,
                                "is_reference": amount == ref_amount,
                                "ref_shift": shift,
                                "distance": dist,
                            }
                        )
                        log.info(f"  {variant_mod_sig}: {dist:.6g} (shift={shift})")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "distances.csv")
    df.to_csv(csv_path, index=False)
    log.info(f"Saved {len(df)} distances to {csv_path}")

    n_plots = 0
    for loss_name, loss_df in df.groupby("loss_fn", sort=False):
        curves = {
            keys: summarize_curve(group)
            for keys, group in loss_df.groupby(["group", "reference"], sort=False)
        }
        # Fixed y range per loss function so its plots can be compared
        ylim = compute_ylim(list(curves.values()))
        log.info(f"{loss_name} ylim = ({ylim[0]:.6g}, {ylim[1]:.6g})")
        for (group_name, mod_sig), curve in curves.items():
            plot_distance_curve(
                curve,
                loss_name,
                group_name,
                mod_sig,
                ylim=ylim,
                max_shift=max_shift if use_rand_phase_shift else 0,
                save_dir=save_dir,
            )
            n_plots += 1
    log.info(f"Saved {n_plots} plots to {save_dir}")
