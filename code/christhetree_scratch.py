import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pyloudnorm as pyln
import torch as tr
import torchaudio
from torch import Tensor as T

from features import Loudness, SpectralCentroid, SpectralFlatness

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def create_wavetable_sweep(wt: T, sr: int = 44100, duration: float = 3.0):
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


def loudness_normalize(audio, sr=44100, target_lufs=-8):
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


def plot_wt(wt: T, name: str) -> None:
    loudness_metric = Loudness(sr)
    centroid_metric = SpectralCentroid(
        sr, window="flat_top", compress=True, floor=1e-4, scaling="kazazis"
    )
    flatness_metric = SpectralFlatness()

    loudness = loudness_metric(wt)
    centroid = centroid_metric(wt)
    flatness = flatness_metric(wt)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(loudness.cpu().numpy())
    plt.title("Loudness")
    plt.xlabel("Frame")
    plt.ylabel("Loudness (in dBFS)")

    plt.subplot(1, 3, 2)
    plt.plot(centroid.cpu().numpy())
    plt.title("Spectral Centroid")
    plt.xlabel("Frame")
    plt.ylabel("Centroid (FFT bin index)")

    plt.subplot(1, 3, 3)
    plt.plot(flatness.cpu().numpy())
    plt.title("Spectral Flatness")
    plt.xlabel("Frame")
    plt.ylabel("Flatness (in dBFS)")

    plt.suptitle(f"{name}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    wt_dir = "../data/ableton/"
    save_dir = "../out/"
    sr = 44100
    sweep_dur_sec = 4.0
    wt_samples = 1024
    chunked_n_pos = int(sweep_dur_sec * sr) // wt_samples

    # wt_name = "basics__harmonic_series__7_1024"
    # wt_name = "basics__fm_fold__78_1024"
    wt_name = "complex__bitkart__149_1024"

    wt_path = os.path.join(wt_dir, f"{wt_name}.pt")
    wt = tr.load(wt_path)
    log.info(f"wt.shape: {wt.shape}")

    sweep = create_wavetable_sweep(wt, sr=sr, duration=sweep_dur_sec)
    sweep = tr.from_numpy(sweep).float()
    save_path = os.path.join(save_dir, f"{wt_name}.wav")
    torchaudio.save(save_path, sweep.unsqueeze(0), sr)
    log.info(f"Saved wavetable sweep to: {save_path}")

    wt_flattened = wt.view(1, -1)
    save_path = os.path.join(save_dir, f"{wt_name}_flattened.wav")
    torchaudio.save(save_path, wt_flattened, sr)
    log.info(f"wt_flattened.shape: {wt_flattened.shape}")
    log.info(f"wt_flattened.abs().max(): {wt_flattened.abs().max()}")
    #
    # wt_flattened = wt.view(-1).numpy()
    # wt_normed, loudness, gain = loudness_normalize(wt_flattened, sr, target_lufs=-12)
    # wt_normed = tr.from_numpy(wt_normed).view(1, -1)
    # log.info(f"wt_normed.shape: {wt_normed.shape}")
    # log.info(f"wt_normed.abs().max(): {wt_normed.abs().max()}")

    chunked_sweep = sweep[: chunked_n_pos * wt_samples]
    chunked_sweep = chunked_sweep.view(-1, wt_samples)
    log.info(f"chunked sweep shape: {chunked_sweep.shape}")

    plot_wt(wt, wt_name)
    plot_wt(chunked_sweep, f"{wt_name} - chunked sweep")
