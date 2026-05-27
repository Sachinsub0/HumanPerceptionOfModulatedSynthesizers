import logging
import os
from typing import Dict, List, Iterator, Optional, Tuple, Union, Any

import numpy as np
import pyloudnorm as pyln
import torch as tr
import torch.nn.functional as F
from scipy.stats import loguniform
from torch import Tensor as T, nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ReadOnlyTensorDict(nn.Module):
    def __init__(self, data: Dict[str | int, T], persistent: bool = True):
        super().__init__()
        self.persistent = persistent
        self.keys = set(data.keys())
        for k, v in data.items():
            self.register_buffer(f"tensor_{k}", v, persistent=persistent)

    def __getitem__(self, key: str | int) -> T:
        return self.get_buffer(f"tensor_{key}")

    def __contains__(self, key: str | int) -> bool:
        return key in self.keys

    def __len__(self) -> int:
        return len(self.keys)

    def __iter__(self) -> Iterator[str | int]:
        return iter(self.keys)

    def keys(self) -> Iterator[str | int]:
        return iter(self.keys)

    def values(self) -> Iterator[T]:
        for k in self.keys:
            yield self[k]

    def items(self) -> Iterator[Tuple[str | int, T]]:
        for k in self.keys:
            yield k, self[k]


def linear_interpolate_last_dim(x: T, n: int, align_corners: bool = True) -> T:
    n_dim = x.ndim
    assert 1 <= n_dim <= 3
    if x.size(-1) == n:
        return x
    if n_dim == 1:
        x = x.view(1, 1, -1)
    elif n_dim == 2:
        x = x.unsqueeze(1)
    x = F.interpolate(x, n, mode="linear", align_corners=align_corners)
    if n_dim == 1:
        x = x.view(-1)
    elif n_dim == 2:
        x = x.squeeze(1)
    return x


def choice(items: List[Any]) -> Any:
    assert len(items) > 0
    idx = randint(0, len(items))
    return items[idx]


def randint(low: int, high: int, n: int = 1) -> Union[int, T]:
    x = tr.randint(low=low, high=high, size=(n,))
    if n == 1:
        return x.item()
    return x


def sample_uniform(low: float, high: float, n: int = 1) -> Union[float, T]:
    x = (tr.rand(n) * (high - low)) + low
    if n == 1:
        return x.item()
    return x


def sample_log_uniform(low: float, high: float, n: int = 1) -> Union[float, T]:
    # TODO(cm): replace with torch
    if low == high:
        if n == 1:
            return low
        else:
            return tr.full(size=(n,), fill_value=low)
    x = loguniform.rvs(low, high, size=n)
    if n == 1:
        return float(x)
    return tr.from_numpy(x)


def create_wavetable_sweep(
    wt: T,
    sr: int = 44100,
    duration: float = 4.0,
    mod_signal: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Render audio by sweeping through wavetable frames.

    wt: tensor [num_frames, frame_length]
    sr: sample rate
    duration: output duration in seconds
    mod_signal: optional modulation signal of length total_samples, values
        in [0, 1] mapped to [0, num_frames-1]. If None, a linear sweep is used.
    """
    wt = wt.detach().cpu().numpy()

    num_frames, frame_len = wt.shape
    total_samples = int(sr * duration)

    if mod_signal is None:
        frame_positions = np.linspace(0, num_frames - 1, total_samples)
    else:
        frame_positions = np.clip(mod_signal, 0.0, 1.0) * (num_frames - 1)

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
