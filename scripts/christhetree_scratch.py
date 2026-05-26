import glob
import logging
import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyloudnorm as pyln
import torch as tr
import torchaudio
from torch import Tensor as T

from features import Loudness, SpectralCentroid, SpectralFlatness
from util import linear_interpolate_last_dim

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

FEATURE_NAMES = [
    "Loudness",
    "Spectral Centroid",
    "Spectral Flatness",
    "Warmth",
    "Richness",
]
FEATURE_YLABELS = [
    "Loudness",
    "Centroid",
    "Flatness",
    "Warmth",
    "Richness",
]


def create_wavetable_sweep(wt: T, sr: int = 44100, duration: float = 3.0) -> np.ndarray:
    """
    Creates a linear sweep through wavetable frames.
    wt: tensor [num_frames, frame_length]
    """
    wt = wt.detach().cpu().numpy()

    num_frames, frame_len = wt.shape
    total_samples = int(sr * duration)

    # Continuous frame position (0 → num_frames-1)
    frame_positions = np.linspace(0, num_frames - 1, total_samples)

    output = np.zeros(total_samples)

    for i, pos in enumerate(frame_positions):
        idx_low = int(np.floor(pos))
        idx_high = min(idx_low + 1, num_frames - 1)
        frac = pos - idx_low

        # Linear interpolation between frames
        frame = (1 - frac) * wt[idx_low] + frac * wt[idx_high]

        # Wrap inside frame length
        sample_index = i % frame_len
        output[i] = frame[sample_index]

    # Normalize to avoid clipping
    # output /= np.max(np.abs(output) + 1e-8)
    if np.abs(output).max() > 1.0:
        log.warning("wavetable sweep is clipping")

    return output


def loudness_normalize(
    audio: np.ndarray, sr: int, target_lufs: float = -16
) -> Tuple[np.ndarray, float, float]:
    """
    Normalize audio to target LUFS.

    audio: numpy array
    sr: sample rate
    target_lufs: desired loudness (e.g. -16, -14, -12)
    """

    meter = pyln.Meter(sr)  # ITU-R BS.1770

    loudness = meter.integrated_loudness(audio)

    # Compute gain
    gain = target_lufs - loudness

    # Apply gain
    normalized_audio = pyln.normalize.loudness(audio, loudness, target_lufs)

    # Optional: clipping protection
    # peak = np.max(np.abs(normalized_audio))
    # if peak > 1.0:
    #     normalized_audio = normalized_audio / peak

    return normalized_audio, loudness, gain


def compute_warmth_curve(frame_batch: T, eps: float = 1e-8) -> T:
    """
    warmth = odd harmonic power ratio (excluding DC)
    """

    fft = tr.fft.rfft(frame_batch)
    power = tr.abs(fft) ** 2

    odd_power = power[:, 1::2].sum(dim=1)
    total_power = power[:, 1:].sum(dim=1)

    warmth = odd_power / (total_power + eps)

    return warmth


def compute_richness_curve(frame_batch: T, eps: float = 1e-8) -> T:
    fft = tr.fft.rfft(frame_batch)
    mag = tr.abs(fft)
    power = mag**2

    freqs = tr.linspace(0, 1, power.shape[1], device=power.device)

    total = power.sum(dim=1, keepdim=True) + eps
    centroid = (power * freqs).sum(dim=1, keepdim=True) / total

    spread = tr.sqrt((power * (freqs - centroid) ** 2).sum(dim=1) / total.squeeze(1))

    k = 7.5
    richness = tr.log(spread * (tr.exp(tr.tensor(k)) - 1) + 1) / k

    return richness


def compute_features(wt: T, sr: int, max_n_pos: int) -> dict[str, T]:
    loudness_metric = Loudness(sr)
    centroid_metric = SpectralCentroid(
        sr, window="flat_top", compress=True, floor=1e-4, scaling="kazazis"
    )
    flatness_metric = SpectralFlatness()

    features = {}
    features["Loudness"] = linear_interpolate_last_dim(loudness_metric(wt), max_n_pos)
    features["Spectral Centroid"] = linear_interpolate_last_dim(
        centroid_metric(wt), max_n_pos
    )
    features["Spectral Flatness"] = linear_interpolate_last_dim(
        flatness_metric(wt), max_n_pos
    )
    features["Warmth"] = linear_interpolate_last_dim(
        compute_warmth_curve(wt), max_n_pos
    )
    features["Richness"] = linear_interpolate_last_dim(
        compute_richness_curve(wt), max_n_pos
    )
    return features


def plot_wt(features: dict[str, T], name: str) -> None:
    plt.figure(figsize=(20, 4))
    for i, (feat_name, ylabel) in enumerate(zip(FEATURE_NAMES, FEATURE_YLABELS)):
        plt.subplot(1, 5, i + 1)
        plt.plot(features[feat_name].cpu().numpy())
        plt.title(feat_name)
        plt.xlabel("Frame")
        plt.ylabel(ylabel)
    plt.suptitle(f"{name}")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(all_features: dict[str, list[np.ndarray]]) -> None:
    stacked = np.column_stack(
        [np.concatenate(all_features[name]) for name in FEATURE_NAMES]
    )
    corr = np.corrcoef(stacked, rowvar=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(FEATURE_NAMES)
    for i in range(len(FEATURE_NAMES)):
        for j in range(len(FEATURE_NAMES)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix (all wavetables)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    wt_dir = "../data/ableton/"
    save_dir = "../out/"
    sr = 44100
    sweep_dur_sec = 4.0
    target_lufs = -16
    wt_samples = 1024
    max_n_pos = 256

    wt_paths = sorted(glob.glob(os.path.join(wt_dir, "*.pt")))
    log.info(f"Found {len(wt_paths)} wavetables in {wt_dir}")

    all_features: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_NAMES}

    for wt_path in wt_paths:
        wt_name = os.path.splitext(os.path.basename(wt_path))[0]
        wt = tr.load(wt_path)
        log.info(f"wt_name: {wt_name}, wt.shape: {wt.shape}")

        features = compute_features(wt, sr, max_n_pos)
        for name in FEATURE_NAMES:
            all_features[name].append(features[name].cpu().numpy())
        # plot_wt(features, wt_name)

        # sweep = create_wavetable_sweep(wt, sr=sr, duration=sweep_dur_sec)
        # sweep_normed, loudness, gain = loudness_normalize(
        #     sweep, sr, target_lufs=target_lufs
        # )
        # log.info(f"loudness: {loudness:.1f} LUFS, gain: {gain:.1f} dB")
        # sweep_normed = tr.from_numpy(sweep_normed).float()
        # save_path = os.path.join(save_dir, f"{wt_name}_{target_lufs}lufs.wav")
        # torchaudio.save(save_path, sweep_normed.unsqueeze(0), sr)
        # log.info(f"Saved normalized sweep to: {save_path}")

        # sweep = tr.from_numpy(sweep).float()
        # for overlap in [0.0, 0.25, 0.5, 0.75]:
        #     hop_size = int(wt_samples * (1 - overlap))
        #     chunked_sweep_sw = sweep.unfold(0, wt_samples, hop_size)
        #     log.info(
        #         f"chunked sweep (overlap={overlap:.0%}) shape: {chunked_sweep_sw.shape}"
        #     )
        #     plot_wt(
        #         chunked_sweep_sw,
        #         f"{wt_name} - chunked sweep (overlap={overlap:.0%})",
        #     )

        # time.sleep(1.0)

    plot_correlation_matrix(all_features)
