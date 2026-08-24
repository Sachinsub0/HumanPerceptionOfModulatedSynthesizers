import glob
import logging
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as tr
import torchaudio
from torch import Tensor as T

from modulations_mod_extraction import make_mod_signal, make_quasi_periodic
from util import create_wavetable_sweep, loudness_normalize

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

SERIES_COLOR = "#2a78d6"
AXIS_COLOR = "#52514e"

# Modulation amounts are displayed as (scale, unit), e.g. an irregularity amount
# of 0.125 is displayed as 12.5%
MOD_SIG_AMOUNT_UNITS = {
    "amp": (1.0, ""),
    "freq": (1.0, " Hz"),
    "reg": (100.0, "%"),
}
# Wide enough for the five modulation signals of one modulation type in a row.
# The height is just enough for the 2:1 plots and their labels, since the width
# is what determines how large the plots end up
FIG_SIZE = (12.0, 1.8)
# Height / width of each plotting area, i.e. each plot is twice as wide as tall
BOX_ASPECT = 0.5
DPI = 300
FONT_SIZE = 12


def make_mod_sig(
    mod_type: str,
    amount: float,
    n_samples: int,
    sr: int,
    mod_freq: float = 1.0,
    amp_center_val: float = 0.5,
    reg_seed: int = 42,
) -> T:
    """Make the modulation signal of one stimulus. amount is the modulation
    amount of mod_type: the rate in Hz for "freq", the depth for "amp", and the
    randomness of the periodicity for "reg". mod_freq is the fixed rate of the
    "amp" and "reg" modulations and is unused for "freq"."""
    if mod_type == "freq":
        return make_mod_signal(n_samples, sr, amount, shape="cos")
    elif mod_type == "amp":
        mod_sig = make_mod_signal(n_samples, sr, mod_freq, shape="cos", phase=tr.pi / 2)
        mod_sig = amp_center_val + (mod_sig - 0.5) * amount
        log.info(
            f"amp={amount:.2f} mod_sig min={mod_sig.min():.4f} "
            f"max={mod_sig.max():.4f} mean={mod_sig.mean():.4f}"
        )
        return mod_sig
    elif mod_type == "reg":
        mod_sig = make_mod_signal(n_samples, sr, mod_freq, shape="cos")
        mod_sig, norm_gaps = make_quasi_periodic(
            mod_sig, randomness=amount, seed=reg_seed
        )
        log.info(f"r={amount:.2f} intervals: {[f'{g:.2f}' for g in norm_gaps]}")
        return mod_sig
    else:
        raise ValueError(f"Unsupported mod_type: {mod_type}")


def format_amounts(
    mod_type: str, amounts: List[float], max_decimals: int = 6
) -> List[str]:
    """Display the modulation amounts of one modulation type with the same
    number of decimals, using the fewest that represent all of them exactly."""
    scale, unit = MOD_SIG_AMOUNT_UNITS[mod_type]
    vals = [a * scale for a in amounts]
    n_decimals = max_decimals
    for d in range(max_decimals + 1):
        if all(abs(v - round(v, d)) < 1e-9 for v in vals):
            n_decimals = d
            break
    return [f"{v:.{n_decimals}f}{unit}" for v in vals]


def plot_mod_signals(
    mod_type: str,
    amounts: List[float],
    n_samples: int,
    sr: int,
    mod_freq: float = 1.0,
    amp_center_val: float = 0.5,
    reg_seed: int = 42,
    save_dir: str = "",
    fig_size: Tuple[float, float] = FIG_SIZE,
    dpi: int = DPI,
) -> None:
    """Plot the modulation signals of one modulation type side by side, in order
    of increasing amount. The lowest amount is the reference of the listening
    test, the rest are labelled amount 1 onwards."""
    amounts = sorted(amounts)
    mod_sigs = [
        make_mod_sig(
            mod_type, amount, n_samples, sr, mod_freq, amp_center_val, reg_seed
        )
        for amount in amounts
    ]
    amount_labels = format_amounts(mod_type, amounts)
    dur_sec = n_samples / sr
    t = np.arange(n_samples) / sr

    # Constrained layout keeps the shared x label tight against the plots
    fig, axs = plt.subplots(
        1,
        len(mod_sigs),
        figsize=fig_size,
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    axs = axs[0]
    for idx, (ax, mod_sig) in enumerate(zip(axs, mod_sigs)):
        ax.set_box_aspect(BOX_ASPECT)  # 2:1 plotting area, not a 2:1 figure
        ax.plot(t, mod_sig.numpy(), color=SERIES_COLOR, linewidth=1.5)
        label = "Reference" if idx == 0 else f"Amount {idx}"
        ax.set_title(f"{label} ({amount_labels[idx]})", fontsize=FONT_SIZE)
        ax.set_xlim(0.0, dur_sec)
        ax.set_xticks(np.arange(0.0, dur_sec + 1e-6, 1.0))
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=FONT_SIZE)
        ax.grid(True, color=AXIS_COLOR, alpha=0.15, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axs[0].set_ylabel("Wavetable position", fontsize=FONT_SIZE)
    fig.supxlabel("Time (s)", fontsize=FONT_SIZE)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.0)
    # plt.show()
    if save_dir:
        save_name = f"mod_sigs__{mod_type}.png"
        plt.savefig(os.path.join(save_dir, save_name), dpi=dpi)
        log.info(f"Saved {save_name} ({len(mod_sigs)} signals, {dpi} dpi)")
    plt.close(fig)


if __name__ == "__main__":
    wavetable_dir = os.path.join("../data/listening_test")
    save_dir = "../out/"
    plot_dir = "../out/mod_sigs"
    sr = 44100
    sweep_dur_sec = 4.0
    target_lufs = -18
    fade_samples = 256

    # freq_vals = []
    freq_vals = [0.25, 0.5, 1.0, 2.0, 4.0]

    amp_freq = 1.0
    amp_center_val = 0.5
    # amp_vals = []
    amp_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    # amp_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    # amp_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    reg_freq = 1.0
    reg_seed = 42
    # reg_vals = []
    reg_vals = [0.0, 0.125, 0.25, 0.375, 0.5]
    # reg_vals = [0.0, 0.2, 0.4, 0.6, 0.8]

    n_samples = int(sr * sweep_dur_sec)
    wt_paths = sorted(glob.glob(os.path.join(wavetable_dir, "*.pt")))
    log.info(f"Found {len(wt_paths)} wavetables")
    fade_in = np.linspace(0.0, 1.0, fade_samples)
    fade_out = np.linspace(1.0, 0.0, fade_samples)

    os.makedirs(plot_dir, exist_ok=True)
    for mod_type, amounts, mod_freq in [
        ("amp", amp_vals, amp_freq),
        ("freq", freq_vals, 1.0),
        ("reg", reg_vals, reg_freq),
    ]:
        plot_mod_signals(
            mod_type,
            amounts,
            n_samples,
            sr,
            mod_freq=mod_freq,
            amp_center_val=amp_center_val,
            reg_seed=reg_seed,
            save_dir=plot_dir,
        )

    for wt_path in wt_paths:
        wt_name = os.path.splitext(os.path.basename(wt_path))[0]
        wt = tr.load(wt_path, weights_only=True)
        lut_path = os.path.join(wavetable_dir, f"{wt_name}__lut.npy")
        if os.path.exists(lut_path):
            lut = np.load(lut_path)
            log.info(
                f"Processing wavetable: {wt_name} (shape={wt.shape}, lut={lut.shape})"
            )
        else:
            lut = None
            log.warning(
                f"No LUT file found at {lut_path}, skipping warping for {wt_name}"
            )

        for freq in freq_vals:
            mod_sig = make_mod_sig("freq", freq, n_samples, sr)
            if lut is not None:
                mod_sig_warped = np.interp(
                    mod_sig.numpy(), np.linspace(0, 1, len(lut)), lut
                )
            else:
                mod_sig_warped = mod_sig.numpy()
            sweep = create_wavetable_sweep(
                wt, sr=sr, duration=sweep_dur_sec, mod_signal=mod_sig_warped
            )
            sweep_norm, loudness, gain = loudness_normalize(sweep, sr, target_lufs)

            sweep_norm[:fade_samples] *= fade_in
            sweep_norm[-fade_samples:] *= fade_out

            save_name = f"{wt_name}__freq_{freq:.2f}hz_{target_lufs}lufs.wav"
            save_path = os.path.join(save_dir, save_name)
            torchaudio.save(
                save_path,
                tr.tensor(sweep_norm).unsqueeze(0).expand(2, -1).float(),
                sr,
            )
            log.info(f"Saved {save_name} (loudness={loudness:.1f}, gain={gain:.1f}dB)")

        for amp in amp_vals:
            mod_sig = make_mod_sig(
                "amp", amp, n_samples, sr, amp_freq, amp_center_val=amp_center_val
            )
            if lut is not None:
                mod_sig_warped = np.interp(
                    mod_sig.numpy(), np.linspace(0, 1, len(lut)), lut
                )
            else:
                mod_sig_warped = mod_sig.numpy()

            sweep = create_wavetable_sweep(
                wt, sr=sr, duration=sweep_dur_sec, mod_signal=mod_sig_warped
            )
            sweep_norm, loudness, gain = loudness_normalize(sweep, sr, target_lufs)

            sweep_norm[:fade_samples] *= fade_in
            sweep_norm[-fade_samples:] *= fade_out

            save_name = (
                f"{wt_name}__amp_{amp_freq:.2f}hz_{amp:.2f}_{target_lufs}lufs.wav"
            )
            save_path = os.path.join(save_dir, save_name)
            torchaudio.save(
                save_path,
                tr.tensor(sweep_norm).unsqueeze(0).expand(2, -1).float(),
                sr,
            )
            log.info(f"Saved {save_name} (loudness={loudness:.1f}, gain={gain:.1f}dB)")

        for reg in reg_vals:
            mod_sig = make_mod_sig(
                "reg", reg, n_samples, sr, reg_freq, reg_seed=reg_seed
            )
            if lut is not None:
                mod_sig_warped = np.interp(
                    mod_sig.numpy(), np.linspace(0, 1, len(lut)), lut
                )
            else:
                mod_sig_warped = mod_sig.numpy()

            sweep = create_wavetable_sweep(
                wt, sr=sr, duration=sweep_dur_sec, mod_signal=mod_sig_warped
            )
            sweep_norm, loudness, gain = loudness_normalize(sweep, sr, target_lufs)

            sweep_norm[:fade_samples] *= fade_in
            sweep_norm[-fade_samples:] *= fade_out

            save_name = (
                f"{wt_name}__reg_{reg_freq:.2f}hz_{reg:.3f}_{target_lufs}lufs.wav"
            )
            save_path = os.path.join(save_dir, save_name)
            torchaudio.save(
                save_path,
                tr.tensor(sweep_norm).unsqueeze(0).expand(2, -1).float(),
                sr,
            )
            log.info(f"Saved {save_name} (loudness={loudness:.1f}, gain={gain:.1f}dB)")
