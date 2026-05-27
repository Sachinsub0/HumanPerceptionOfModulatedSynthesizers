import glob
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as tr
import torchaudio
from torch import Tensor as T

from features import Loudness, SpectralCentroid, SpectralFlatness
from util import linear_interpolate_last_dim, create_wavetable_sweep, loudness_normalize

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

FEATURE_NAMES = [
    "Loudness",
    "Spectral Flatness",
    "Spectral Centroid",
    "Warmth",
    "Richness",
]
FEATURE_YLABELS = [
    "Loudness",
    "Flatness",
    "Centroid",
    "Warmth",
    "Richness",
]


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


def compute_features(wt: T, sr: int, max_n_pos: int) -> Dict[str, T]:
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


def plot_wt(
    features: Dict[str, T],
    name: str,
    save_dir: str = "",
    z_stats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    plt.figure(figsize=(20, 4))
    for i, (feat_name, ylabel) in enumerate(zip(FEATURE_NAMES, FEATURE_YLABELS)):
        plt.subplot(1, 5, i + 1)
        vals = features[feat_name].cpu().numpy()
        if z_stats is not None:
            mu, std = z_stats[feat_name]
            vals = (vals - mu) / (std + 1e-8)
            plt.ylim(-3, 3)
            ylabel = "z-score"
        plt.plot(vals)
        plt.title(feat_name)
        plt.xlabel("Frame")
        plt.ylabel(ylabel)
    plt.suptitle(f"{name}")
    plt.tight_layout()
    if save_dir:
        suffix = "_features__z" if z_stats is not None else "_features"
        plt.savefig(os.path.join(save_dir, f"{name}{suffix}.png"), dpi=150)
    plt.close()


def rank_wavetables(
    all_features: Dict[str, List[np.ndarray]],
    wt_names: List[str],
    range_metric: str,
    low_corr_metrics: List[str],
) -> List[Tuple[str, float, float, float]]:
    n_wt = len(wt_names)
    results = []
    for i in range(n_wt):
        target = all_features[range_metric][i]
        feat_range = target.max() - target.min()

        abs_corrs = {}
        for metric in low_corr_metrics:
            other = all_features[metric][i]
            r = np.corrcoef(target, other)[0, 1]
            abs_corrs[metric] = abs(r)
        mean_abs_corr = np.mean(list(abs_corrs.values()))

        score = feat_range * (1.0 - mean_abs_corr)
        results.append((wt_names[i], score, feat_range, abs_corrs))

    results.sort(key=lambda x: x[1], reverse=True)

    log.info(
        f"Ranking by: high {range_metric} range, low correlation with {low_corr_metrics}"
    )
    for rank, (name, score, feat_range, abs_corrs) in enumerate(results):
        corr_str = "  ".join(
            f"|corr({metric})|={abs_corrs[metric]:.4f}" for metric in low_corr_metrics
        )
        log.info(
            f"  {rank + 1:3d}. {name:40s}  score={score:.4f}  "
            f"range={feat_range:.4f}  {corr_str}"
        )
    return results


def rank_wavetables_by_range(
    all_features: Dict[str, List[np.ndarray]],
    wt_names: List[str],
    high_range_metric: str,
    low_range_metrics: List[str],
) -> List[Tuple[str, float, Dict[str, float]]]:
    all_metrics = [high_range_metric] + low_range_metrics
    n_wt = len(wt_names)

    stats = ["range", "min", "max"]
    raw: Dict[str, Dict[str, np.ndarray]] = {}
    for metric in all_metrics:
        vals = [all_features[metric][i] for i in range(n_wt)]
        raw[metric] = {
            "range": np.array([v.max() - v.min() for v in vals]),
            "min": np.array([v.min() for v in vals]),
            "max": np.array([v.max() for v in vals]),
        }

    z: Dict[str, Dict[str, np.ndarray]] = {}
    for metric in all_metrics:
        z[metric] = {}
        for stat in stats:
            mu = raw[metric][stat].mean()
            std = raw[metric][stat].std()
            z[metric][stat] = (raw[metric][stat] - mu) / (std + 1e-8)

    results = []
    for i in range(n_wt):
        z_high = z[high_range_metric]["range"][i]
        z_lows = np.array([z[m]["range"][i] for m in low_range_metrics])
        score = z_high - z_lows.mean()
        per_metric = {
            m: {stat: float(z[m][stat][i]) for stat in stats}
            for m in all_metrics
        }
        results.append((wt_names[i], float(score), per_metric))

    results.sort(key=lambda x: x[1], reverse=True)

    log.info(
        f"Ranking by: high {high_range_metric} z-range, low z-range in {low_range_metrics}"
    )
    for rank, (name, score, per_metric) in enumerate(results):
        parts = []
        for m in all_metrics:
            parts.append(
                f"{m}: z_range={per_metric[m]['range']:+.2f} "
                f"z_min={per_metric[m]['min']:+.2f} "
                f"z_max={per_metric[m]['max']:+.2f}"
            )
        log.info(
            f"  {rank + 1:3d}. {name:40s}  score={score:+.2f}  " + "  |  ".join(parts)
        )
    return results


def plot_correlation_matrix(
    feature_arrays: List[np.ndarray], name: str, save_dir: str = ""
) -> None:
    stacked = np.column_stack(feature_arrays)
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
    ax.set_title(name)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{name}_corr.png"), dpi=150)
    plt.close()


def create_synthetic_wavetable(
    n_pos: int = 256,
    n_samples: int = 1024,
    centroids: np.ndarray = None,
    sigmas: np.ndarray = None,
    warmths: np.ndarray = None,
    centroid_correction: bool = False,
    seed: int = 42,
) -> T:
    """
    Create a synthetic wavetable by constructing each frame in the
    frequency domain with a Gaussian spectral envelope.

    Per-position arrays control which parameter sweeps and which stays fixed:
    - centroids: Gaussian center per position (FFT bin index)
    - sigmas: Gaussian width per position (bins)
    - warmths: odd-to-total power ratio per position

    Each frame:
    1. Gaussian envelope at (centroids[p], sigmas[p]) sets spectral shape.
    2. Even harmonics are rescaled to enforce warmths[p].
    3. Fixed random phases (seeded) are shared across all positions.
    4. Peak-normalized to [-1, 1].

    centroid_correction: if True, iteratively shifts the Gaussian center
        to compensate for asymmetric clipping at bin 0.
    """
    n_bins = n_samples // 2 + 1
    bins = np.arange(n_bins, dtype=np.float64)

    if centroids is None:
        centroids = np.full(n_pos, 40.0)
    if sigmas is None:
        sigmas = np.full(n_pos, 20.0)
    if warmths is None:
        warmths = np.full(n_pos, 0.5)

    rng = np.random.RandomState(seed)
    phases = rng.uniform(0, 2 * np.pi, n_bins)
    phases[0] = 0.0

    odd_mask = np.zeros(n_bins, dtype=bool)
    even_mask = np.zeros(n_bins, dtype=bool)
    odd_mask[1::2] = True
    even_mask[2::2] = True

    frames = np.zeros((n_pos, n_samples))
    for p in range(n_pos):
        center = centroids[p]
        if centroid_correction:
            for _ in range(10):
                envelope = np.exp(-0.5 * ((bins - center) / sigmas[p]) ** 2)
                envelope[0] = 0.0
                power = envelope ** 2
                actual = (bins * power).sum() / (power.sum() + 1e-12)
                center += centroids[p] - actual
        envelope = np.exp(-0.5 * ((bins - center) / sigmas[p]) ** 2)
        envelope[0] = 0.0

        odd_power = (envelope[odd_mask] ** 2).sum()
        even_power = (envelope[even_mask] ** 2).sum()

        # Scale even harmonics to enforce warmth:
        # warmth = odd / (odd + s^2 * even) = w  =>  s = sqrt(odd*(1-w) / (w*even))
        w = warmths[p]
        if even_power > 0 and odd_power > 0:
            even_scale = np.sqrt(odd_power * (1 - w) / (w * even_power))
        else:
            even_scale = 1.0
        amplitudes = envelope.copy()
        amplitudes[even_mask] *= even_scale

        spectrum = amplitudes * np.exp(1j * phases)
        frame = np.fft.irfft(spectrum, n=n_samples)

        peak = np.abs(frame).max()
        if peak > 0:
            frame /= peak
        frames[p] = frame

    return tr.from_numpy(frames).float()


def create_synthetic_centroid_sweep(
    n_pos: int = 256,
    n_samples: int = 1024,
    centroid_range: Tuple[float, float] = (10.0, 110.0),
    sigma: float = 10.0,
    target_warmth: float = 0.5,
    seed: int = 42,
) -> T:
    return create_synthetic_wavetable(
        n_pos=n_pos,
        n_samples=n_samples,
        centroids=np.geomspace(centroid_range[0], centroid_range[1], n_pos),
        sigmas=np.full(n_pos, sigma),
        warmths=np.full(n_pos, target_warmth),
        seed=seed,
    )


def create_synthetic_warmth_sweep(
    n_pos: int = 256,
    n_samples: int = 1024,
    target_centroid: float = 40.0,
    sigma: float = 20.0,
    warmth_range: Tuple[float, float] = (0.001, 0.999),
    seed: int = 42,
) -> T:
    return create_synthetic_wavetable(
        n_pos=n_pos,
        n_samples=n_samples,
        centroids=np.full(n_pos, target_centroid),
        sigmas=np.full(n_pos, sigma),
        warmths=np.linspace(warmth_range[0], warmth_range[1], n_pos),
        seed=seed,
    )


def create_synthetic_richness_sweep(
    n_pos: int = 256,
    n_samples: int = 1024,
    target_centroid: float = 40.0,
    sigma_range: Tuple[float, float] = (2.0, 40.0),
    target_warmth: float = 0.5,
    seed: int = 42,
) -> T:
    return create_synthetic_wavetable(
        n_pos=n_pos,
        n_samples=n_samples,
        centroids=np.full(n_pos, target_centroid),
        sigmas=np.geomspace(sigma_range[0], sigma_range[1], n_pos),
        warmths=np.full(n_pos, target_warmth),
        centroid_correction=True,
        seed=seed,
    )


if __name__ == "__main__":
    wt_dir = "../data/ableton/"
    save_dir = "../out/"
    sr = 44100
    sweep_dur_sec = 4.0
    target_lufs = -18
    wt_samples = 1024
    max_n_pos = 256

    wt_paths = sorted(glob.glob(os.path.join(wt_dir, "*.pt")))
    log.info(f"Found {len(wt_paths)} wavetables in {wt_dir}")

    all_features: Dict[str, List[np.ndarray]] = {name: [] for name in FEATURE_NAMES}
    wt_names: List[str] = []

    for wt_path in wt_paths:
        wt_name = os.path.splitext(os.path.basename(wt_path))[0]
        wt_names.append(wt_name)
        wt = tr.load(wt_path)
        log.info(f"wt_name: {wt_name}, wt.shape: {wt.shape}")

        features = compute_features(wt, sr, max_n_pos)
        for name in FEATURE_NAMES:
            all_features[name].append(features[name].cpu().numpy())

        # plot_wt(features, wt_name, save_dir)
        # plot_correlation_matrix(
        #     [features[name].cpu().numpy() for name in FEATURE_NAMES],
        #     wt_name,
        #     save_dir,
        # )

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

    # z_stats: Dict[str, Tuple[float, float]] = {}
    # for feat_name in FEATURE_NAMES:
    #     concat = np.concatenate(all_features[feat_name])
    #     z_stats[feat_name] = (float(concat.mean()), float(concat.std()))
    #
    # for i, wt_name in enumerate(wt_names):
    #     features = {
    #         name: tr.from_numpy(all_features[name][i]) for name in FEATURE_NAMES
    #     }
    #     plot_wt(features, wt_name, save_dir, z_stats=z_stats)

    # plot_correlation_matrix(
    #     [np.concatenate(all_features[name]) for name in FEATURE_NAMES],
    #     name="all_wavetables",
    #     save_dir=save_dir,
    # )
    # rank_wavetables(
    #     all_features,
    #     wt_names,
    #     # range_metric="Spectral Centroid",
    #     # low_corr_metrics=["Warmth", "Richness"],
    #     # range_metric="Warmth",
    #     # low_corr_metrics=["Spectral Centroid", "Richness"],
    #     range_metric="Richness",
    #     low_corr_metrics=["Warmth", "Spectral Centroid"],
    # )
    # rank_wavetables_by_range(
    #     all_features,
    #     wt_names,
    #     # high_range_metric="Spectral Centroid",
    #     # low_range_metrics=["Warmth", "Richness"],
    #     # high_range_metric="Warmth",
    #     # low_range_metrics=["Spectral Centroid", "Richness"],
    #     high_range_metric="Richness",
    #     low_range_metrics=["Warmth", "Spectral Centroid"],
    # )

    z_stats: Dict[str, Tuple[float, float]] = {}
    for feat_name in FEATURE_NAMES:
        concat = np.concatenate(all_features[feat_name])
        z_stats[feat_name] = (float(concat.mean()), float(concat.std()))

    new_wt_dir = "../data/listening_test/"
    new_wt_paths = sorted(glob.glob(os.path.join(new_wt_dir, "*.pt")))
    log.info(f"Found {len(new_wt_paths)} wavetables in {new_wt_dir}")

    for wt_path in new_wt_paths:
        wt_name = os.path.splitext(os.path.basename(wt_path))[0]
        wt = tr.load(wt_path, weights_only=True)
        log.info(f"wt_name: {wt_name}, wt.shape: {wt.shape}")

        features = compute_features(wt, sr, max_n_pos)

        plot_wt(features, wt_name, save_dir)
        plot_wt(features, wt_name, save_dir, z_stats=z_stats)
        plot_correlation_matrix(
            [features[name].cpu().numpy() for name in FEATURE_NAMES],
            wt_name,
            save_dir,
        )

        sweep = create_wavetable_sweep(wt, sr=sr, duration=sweep_dur_sec)
        sweep_norm, loudness, gain = loudness_normalize(
            sweep, sr, target_lufs=target_lufs
        )
        log.info(f"{wt_name} loudness: {loudness:.1f} LUFS, gain: {gain:.1f} dB")
        wav_path = os.path.join(save_dir, f"{wt_name}_{target_lufs}lufs.wav")
        torchaudio.save(wav_path, tr.from_numpy(sweep_norm).unsqueeze(0).float(), sr)
        log.info(f"Saved: {wav_path}")

    # synth_sweeps = [
    #     ("synthetic_centroid_sweep", create_synthetic_centroid_sweep()),
    #     ("synthetic_warmth_sweep", create_synthetic_warmth_sweep()),
    #     ("synthetic_richness_sweep", create_synthetic_richness_sweep()),
    # ]
    # for synth_name, synth_wt in synth_sweeps:
    #     pt_path = os.path.join(save_dir, f"{synth_name}__256_1024.pt")
    #     tr.save(synth_wt, pt_path)
    #     log.info(f"Saved: {pt_path}")
    #
    #     synth_features = compute_features(synth_wt, sr, max_n_pos)
    #     plot_wt(synth_features, synth_name, save_dir)
    #     plot_wt(synth_features, synth_name, save_dir, z_stats=z_stats)
    #
    #     sweep = create_wavetable_sweep(synth_wt, sr=sr, duration=sweep_dur_sec)
    #     sweep_normed, loudness, gain = loudness_normalize(
    #         sweep, sr, target_lufs=target_lufs
    #     )
    #     log.info(f"{synth_name} loudness: {loudness:.1f} LUFS, gain: {gain:.1f} dB")
    #     sweep_t = tr.from_numpy(sweep_normed).float()
    #     wav_path = os.path.join(save_dir, f"{synth_name}_{target_lufs}lufs.wav")
    #     torchaudio.save(wav_path, sweep_t.unsqueeze(0), sr)
    #     log.info(f"Saved: {wav_path}")


# INFO:__main__:Ranking by: high Spectral Centroid z-range, low z-range in ['Warmth', 'Richness']
# INFO:__main__:    1. basics__fm_harmonics__256_1024            score=+2.74  Spectral Centroid: z_range=+2.95 z_min=-1.86 z_max=+0.70  |  Warmth: z_range=-1.16 z_min=+2.00 z_max=+0.87  |  Richness: z_range=+1.58 z_min=-1.28 z_max=+0.20
# INFO:__main__:    2. harmonics__strong_seventh__63_1024        score=+2.30  Spectral Centroid: z_range=+1.83 z_min=-1.28 z_max=+0.24  |  Warmth: z_range=-1.19 z_min=+1.98 z_max=+0.80  |  Richness: z_range=+0.25 z_min=-1.01 z_max=-0.98
# INFO:__main__:    3. basics__saw_pw_detune__248_1024           score=+2.29  Spectral Centroid: z_range=+0.96 z_min=-0.43 z_max=+0.49  |  Warmth: z_range=-1.38 z_min=+1.32 z_max=-0.11  |  Richness: z_range=-1.28 z_min=+0.84 z_max=-0.41
# INFO:__main__:    4. basics__basic_shapes__4_1024              score=+2.26  Spectral Centroid: z_range=+2.99 z_min=-1.69 z_max=+1.00  |  Warmth: z_range=-0.80 z_min=+1.64 z_max=+0.87  |  Richness: z_range=+2.26 z_min=-1.89 z_max=+0.22
# INFO:__main__:    5. basics__fm_feedback__255_1024             score=+2.21  Spectral Centroid: z_range=+1.97 z_min=-1.74 z_max=-0.28  |  Warmth: z_range=-1.09 z_min=+1.93 z_max=+0.87  |  Richness: z_range=+0.60 z_min=-1.89 z_max=-1.70
# INFO:__main__:    6. vintage__miniwaves__6_1024                score=+1.94  Spectral Centroid: z_range=+2.43 z_min=-1.54 z_max=+0.56  |  Warmth: z_range=-0.29 z_min=+1.10 z_max=+0.86  |  Richness: z_range=+1.27 z_min=-1.20 z_max=-0.05
# INFO:__main__:    7. basics__sub_1__129_1024                   score=+1.90  Spectral Centroid: z_range=+1.80 z_min=-1.67 z_max=-0.38  |  Warmth: z_range=-1.11 z_min=+1.95 z_max=+0.87  |  Richness: z_range=+0.91 z_min=-1.78 z_max=-1.19
# INFO:__main__:    8. filter__dark_throaty__254_1024            score=+1.66  Spectral Centroid: z_range=+1.03 z_min=-1.24 z_max=-0.64  |  Warmth: z_range=-1.32 z_min=+2.04 z_max=+0.73  |  Richness: z_range=+0.06 z_min=-0.76 z_max=-0.89
# INFO:__main__:    9. harmonics__synced_sines__256_1024         score=+1.63  Spectral Centroid: z_range=+0.71 z_min=-1.61 z_max=-1.58  |  Warmth: z_range=-0.55 z_min=+0.30 z_max=-0.28  |  Richness: z_range=-1.28 z_min=-1.11 z_max=-2.86
# INFO:__main__:   10. filter__modern_sweep_1__21_1024           score=+1.57  Spectral Centroid: z_range=+2.25 z_min=-1.24 z_max=+0.80  |  Warmth: z_range=-0.32 z_min=+0.96 z_max=+0.68  |  Richness: z_range=+1.69 z_min=-1.01 z_max=+0.66

# INFO:__main__:Ranking by: high Spectral Centroid range, low correlation with ['Warmth', 'Richness']
# INFO:__main__:    1. vintage__miniwaves__6_1024                score=6.2987  range=6.3381  |corr(Warmth)|=0.0001  |corr(Richness)|=0.0123
# INFO:__main__:    2. harmonics__sines_bunch__100_1024          score=3.9026  range=4.9000  |corr(Warmth)|=0.3712  |corr(Richness)|=0.0359
# INFO:__main__:    3. complex__dubstep_organ__64_1024           score=3.8134  range=7.7297  |corr(Warmth)|=0.0288  |corr(Richness)|=0.9845
# INFO:__main__:    4. complex__void__256_1024                   score=2.6869  range=6.7828  |corr(Warmth)|=0.2449  |corr(Richness)|=0.9628
# INFO:__main__:    5. vintage__sub3_shapes__167_1024            score=2.4674  range=4.1352  |corr(Warmth)|=0.1893  |corr(Richness)|=0.6174
# INFO:__main__:    6. distortion__phased__178_1024              score=2.4517  range=5.7106  |corr(Warmth)|=0.2215  |corr(Richness)|=0.9198
# INFO:__main__:    7. vintage__bs_sync__228_1024                score=2.3965  range=5.0874  |corr(Warmth)|=0.0880  |corr(Richness)|=0.9699
# INFO:__main__:    8. harmonics__synced_sines__256_1024         score=2.1582  range=3.3272  |corr(Warmth)|=0.1763  |corr(Richness)|=0.5264
# INFO:__main__:    9. collection__violet__40_1024               score=2.0293  range=6.2730  |corr(Warmth)|=0.4962  |corr(Richness)|=0.8568
# INFO:__main__:   10. basics__quad_saw__119_1024                score=1.9477  range=4.3141  |corr(Warmth)|=0.2448  |corr(Richness)|=0.8523


# INFO:__main__:Ranking by: high Warmth z-range, low z-range in ['Spectral Centroid', 'Richness']
# INFO:__main__:    1. basics__pulse_dual__256_1024              score=+2.96  Warmth: z_range=+1.85 z_min=-1.11 z_max=+0.85  |  Spectral Centroid: z_range=-1.02 z_min=+1.05 z_max=+0.36  |  Richness: z_range=-1.20 z_min=+1.22 z_max=+0.17
# INFO:__main__:    2. vintage__logue_saw__166_1024              score=+2.82  Warmth: z_range=+1.88 z_min=-1.13 z_max=+0.87  |  Spectral Centroid: z_range=-0.70 z_min=+0.91 z_max=+0.54  |  Richness: z_range=-1.16 z_min=+1.26 z_max=+0.25
# INFO:__main__:    3. harmonics__biharmonic_steps__70_1024      score=+2.74  Warmth: z_range=+1.89 z_min=-1.13 z_max=+0.87  |  Spectral Centroid: z_range=-0.23 z_min=-1.50 z_max=-2.53  |  Richness: z_range=-1.47 z_min=-0.69 z_max=-2.56
# INFO:__main__:    4. retro__harmonics_4__184_1024              score=+2.59  Warmth: z_range=+1.85 z_min=-1.12 z_max=+0.83  |  Spectral Centroid: z_range=-0.83 z_min=+1.07 z_max=+0.63  |  Richness: z_range=-0.65 z_min=+0.16 z_max=-0.55
# INFO:__main__:    5. vintage__logue_saw_dist__205_1024         score=+2.51  Warmth: z_range=+1.74 z_min=-1.13 z_max=+0.71  |  Spectral Centroid: z_range=-0.35 z_min=+0.36 z_max=+0.13  |  Richness: z_range=-1.19 z_min=+0.43 z_max=-0.82
# INFO:__main__:    6. formant__crispy_form__16_1024             score=+2.48  Warmth: z_range=+1.64 z_min=-0.88 z_max=+0.86  |  Spectral Centroid: z_range=-0.90 z_min=+0.67 z_max=-0.05  |  Richness: z_range=-0.78 z_min=-0.24 z_max=-1.19
# INFO:__main__:    7. harmonics__transistor_square__256_1024    score=+2.41  Warmth: z_range=+1.69 z_min=-1.13 z_max=+0.65  |  Spectral Centroid: z_range=-0.94 z_min=+1.37 z_max=+0.95  |  Richness: z_range=-0.51 z_min=+1.63 z_max=+1.48
# INFO:__main__:    8. retro__echoes__185_1024                   score=+2.30  Warmth: z_range=+1.86 z_min=-1.10 z_max=+0.87  |  Spectral Centroid: z_range=-0.72 z_min=+0.99 z_max=+0.63  |  Richness: z_range=-0.16 z_min=-0.18 z_max=-0.41
# INFO:__main__:    9. formant__riser__87_1024                   score=+2.24  Warmth: z_range=+1.30 z_min=-1.11 z_max=+0.24  |  Spectral Centroid: z_range=-0.87 z_min=+0.58 z_max=-0.17  |  Richness: z_range=-1.01 z_min=+0.71 z_max=-0.27
# INFO:__main__:   10. complex__octa_phase__117_1024             score=+2.15  Warmth: z_range=+1.58 z_min=-0.87 z_max=+0.80  |  Spectral Centroid: z_range=-0.85 z_min=+0.92 z_max=+0.39  |  Richness: z_range=-0.30 z_min=+0.08 z_max=-0.24

# INFO:__main__:Ranking by: high Warmth range, low correlation with ['Spectral Centroid', 'Richness']
# INFO:__main__:    1. harmonics__biharmonic_steps__70_1024      score=0.9623  range=1.0000  |corr(Spectral Centroid)|=0.0261  |corr(Richness)|=0.0492
# INFO:__main__:    2. vintage__bs_sync__228_1024                score=0.9084  range=0.9637  |corr(Spectral Centroid)|=0.0880  |corr(Richness)|=0.0269
# INFO:__main__:    3. retro__harmonics_4__184_1024              score=0.8987  range=0.9890  |corr(Spectral Centroid)|=0.0611  |corr(Richness)|=0.1214
# INFO:__main__:    4. distortion__clipped_sweep__64_1024        score=0.7496  range=0.8485  |corr(Spectral Centroid)|=0.1165  |corr(Richness)|=0.1165
# INFO:__main__:    5. complex__bitten_sync__255_1024            score=0.7435  range=0.8216  |corr(Spectral Centroid)|=0.0488  |corr(Richness)|=0.1414
# INFO:__main__:    6. complex__dubstep_organ__64_1024           score=0.7061  range=0.7173  |corr(Spectral Centroid)|=0.0288  |corr(Richness)|=0.0023
# INFO:__main__:    7. harmonics__sines_bunch__100_1024          score=0.6857  range=0.9082  |corr(Spectral Centroid)|=0.3712  |corr(Richness)|=0.1189
# INFO:__main__:    8. retro__harmonics_3__185_1024              score=0.6726  range=0.7554  |corr(Spectral Centroid)|=0.1141  |corr(Richness)|=0.1051
# INFO:__main__:    9. vintage__jx10_sync__127_1024              score=0.6684  range=0.8475  |corr(Spectral Centroid)|=0.2384  |corr(Richness)|=0.1844
# INFO:__main__:   10. distortion__phased__178_1024              score=0.6468  range=0.7494  |corr(Spectral Centroid)|=0.2215  |corr(Richness)|=0.0523


# INFO:__main__:Ranking by: high Richness z-range, low z-range in ['Warmth', 'Spectral Centroid']
# INFO:__main__:    1. vintage__ob6_shapes__71_1024              score=+1.90  Richness: z_range=+0.66 z_min=-0.15 z_max=+0.57  |  Warmth: z_range=-1.63 z_min=-1.13 z_max=-3.01  |  Spectral Centroid: z_range=-0.85 z_min=+1.23 z_max=+0.85
# INFO:__main__:    2. harmonics__sines_1__16_1024               score=+1.83  Richness: z_range=+1.45 z_min=-1.64 z_max=-0.41  |  Warmth: z_range=-1.08 z_min=+1.92 z_max=+0.87  |  Spectral Centroid: z_range=+0.33 z_min=-1.29 z_max=-1.55
# INFO:__main__:    3. filter__jup_sweep__78_1024                score=+1.53  Richness: z_range=+1.62 z_min=-0.88 z_max=+0.74  |  Warmth: z_range=-1.38 z_min=-1.11 z_max=-2.71  |  Spectral Centroid: z_range=+1.55 z_min=-0.70 z_max=+0.78
# INFO:__main__:    4. basics__sub_2__255_1024                   score=+1.46  Richness: z_range=+1.80 z_min=-1.67 z_max=-0.04  |  Warmth: z_range=-0.42 z_min=+1.24 z_max=+0.87  |  Spectral Centroid: z_range=+1.11 z_min=-1.56 z_max=-1.03
# INFO:__main__:    5. harmonics__spectral_2__24_1024            score=+1.44  Richness: z_range=+2.53 z_min=-1.86 z_max=+0.55  |  Warmth: z_range=+0.52 z_min=+0.28 z_max=+0.87  |  Spectral Centroid: z_range=+1.68 z_min=-1.69 z_max=-0.56
# INFO:__main__:    6. harmonics__spectral_1__24_1024            score=+1.38  Richness: z_range=+0.92 z_min=-0.97 z_max=-0.17  |  Warmth: z_range=-0.38 z_min=-0.12 z_max=-0.55  |  Spectral Centroid: z_range=-0.53 z_min=+0.06 z_max=-0.54
# INFO:__main__:    7. harmonics__spectral_3__8_1024             score=+1.35  Richness: z_range=+2.41 z_min=-1.87 z_max=+0.41  |  Warmth: z_range=+0.44 z_min=+0.36 z_max=+0.87  |  Spectral Centroid: z_range=+1.68 z_min=-1.68 z_max=-0.54
# INFO:__main__:    8. harmonics__sines_2__8_1024                score=+1.33  Richness: z_range=+0.58 z_min=-0.35 z_max=+0.22  |  Warmth: z_range=-1.33 z_min=-1.12 z_max=-2.67  |  Spectral Centroid: z_range=-0.18 z_min=-0.29 z_max=-0.65
# INFO:__main__:    9. filter__acid_saw__46_1024                 score=+1.32  Richness: z_range=+0.12 z_min=-0.99 z_max=-1.10  |  Warmth: z_range=-1.62 z_min=+0.66 z_max=-1.09  |  Spectral Centroid: z_range=-0.76 z_min=-1.80 z_max=-3.61
# INFO:__main__:   10. complex__void__256_1024                   score=+1.29  Richness: z_range=+2.78 z_min=-0.77 z_max=+2.22  |  Warmth: z_range=+0.32 z_min=-0.73 z_max=-0.44  |  Spectral Centroid: z_range=+2.68 z_min=-1.24 z_max=+1.31

# INFO:__main__:Ranking by: high Richness range, low correlation with ['Warmth', 'Spectral Centroid']
# INFO:__main__:    1. harmonics__sines_bunch__100_1024          score=0.3999  range=0.4334  |corr(Warmth)|=0.1189  |corr(Spectral Centroid)|=0.0359
# INFO:__main__:    2. complex__dubstep_organ__64_1024           score=0.3457  range=0.6824  |corr(Warmth)|=0.0023  |corr(Spectral Centroid)|=0.9845
# INFO:__main__:    3. vintage__bs_sync__228_1024                score=0.3012  range=0.6005  |corr(Warmth)|=0.0269  |corr(Spectral Centroid)|=0.9699
# INFO:__main__:    4. distortion__phased__178_1024              score=0.2706  range=0.5266  |corr(Warmth)|=0.0523  |corr(Spectral Centroid)|=0.9198
# INFO:__main__:    5. complex__void__256_1024                   score=0.2601  range=0.6485  |corr(Warmth)|=0.2351  |corr(Spectral Centroid)|=0.9628
# INFO:__main__:    6. collection__violet__40_1024               score=0.2583  range=0.5911  |corr(Warmth)|=0.2691  |corr(Spectral Centroid)|=0.8568
# INFO:__main__:    7. vintage__miniwaves__6_1024                score=0.2518  range=0.4190  |corr(Warmth)|=0.7856  |corr(Spectral Centroid)|=0.0123
# INFO:__main__:    8. complex__bitten_sync__255_1024            score=0.2328  range=0.3271  |corr(Warmth)|=0.1414  |corr(Spectral Centroid)|=0.4353
# INFO:__main__:    9. harmonics__spectral_2__24_1024            score=0.2304  range=0.6105  |corr(Warmth)|=0.2813  |corr(Spectral Centroid)|=0.9637
# INFO:__main__:   10. collection__copper__17_1024               score=0.2017  range=0.3383  |corr(Warmth)|=0.0741  |corr(Spectral Centroid)|=0.7336
