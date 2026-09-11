import logging
import os
from typing import List, Sequence, Tuple, Union

import pandas as pd
import torch as tr
import torchaudio
from auraloss.freq import MultiResolutionSTFTLoss
from torch import Tensor as T
from torch import nn

from losses import (
    Scat1DLoss,
    PANNsEmbeddingLoss,
    ClapEmbeddingLoss,
    MFCCDistance,
    LogMSSLoss,
    JTFSTLoss,
)
from plot_distances import DEFAULT_WAVETABLES, resolve_group
from util import find_variants, parse_amount

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


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


def get_unique_wavetables(entries: Sequence[Union[str, Sequence[str]]]) -> List[str]:
    """Extract ordered list of unique individual wavetables from wavetable/group definitions."""
    unique: List[str] = []
    for entry in entries:
        if isinstance(entry, str):
            if entry not in unique:
                unique.append(entry)
        else:
            for item in entry:
                if item not in unique:
                    unique.append(item)
    return unique


def compute_distances(
    loss_fns: Sequence[Union[nn.Module, Tuple[str, nn.Module]]],
    wavetables: Sequence[str],
    mod_sig_references: Sequence[str],
    samples_dir: str,
    save_path: str,
    sr: int = 44100,
    target_lufs: int = -18,
    use_rand_phase_shift: bool = False,
    max_shift: int = 2048,
    shift_seed: int = 42,
) -> pd.DataFrame:
    """Compute distances for single wavetables and save the result to a TSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    suffix = f"_{target_lufs}lufs.wav"
    rand_gen = tr.Generator().manual_seed(shift_seed)

    loss_entries = [resolve_loss_fn(entry) for entry in loss_fns]
    loss_names = [name for name, _ in loss_entries]
    assert len(set(loss_names)) == len(
        loss_names
    ), f"Loss function names must be unique, got {loss_names}"

    rows = []
    for loss_name, loss_fn in loss_entries:
        for wt_name in wavetables:
            for mod_sig in mod_sig_references:
                ref_path = os.path.join(samples_dir, f"{wt_name}__{mod_sig}{suffix}")
                assert os.path.exists(ref_path), f"Missing reference {ref_path}"
                ref_audio = load_audio(ref_path, sr)
                _, ref_amount, _ = parse_amount(mod_sig)

                variant_paths = find_variants(samples_dir, wt_name, mod_sig, suffix)
                log.info(
                    f"{loss_name} | {wt_name} | {mod_sig}: found {len(variant_paths)} samples"
                )
                for variant_path in variant_paths:
                    variant_name = os.path.basename(variant_path)[: -len(suffix)]
                    variant_mod_sig = variant_name[len(f"{wt_name}__") :]
                    _, amount, _ = parse_amount(variant_mod_sig)
                    audio = load_audio(variant_path, sr)
                    assert (
                        audio.shape == ref_audio.shape
                    ), f"Shape mismatch: {audio.shape} vs {ref_audio.shape}"
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
    df.to_csv(save_path, index=False, sep="\t")
    log.info(f"Saved {len(df)} distances to {save_path}")
    return df


if __name__ == "__main__":
    samples_dir = "../out/audio_samples"
    save_dir = "../out/distances"
    tsv_path = os.path.join(save_dir, "distances_testing.tsv")
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
        # ("jtfs2", JTFSTLoss(shape=176400, J=12, Q1=8, Q2=2, J_fr=5, Q_fr=2, T=2048, F=1, format_="joint", p=2, use_rho_log1p=True)),
    ]
    wavetables = DEFAULT_WAVETABLES
    mod_sig_references = [
        "amp_1.00hz_0.10",
        "freq_0.25hz",
        "reg_1.00hz_0.000",
    ]

    os.makedirs(save_dir, exist_ok=True)

    # 1. Resolve group definitions
    groups = [resolve_group(entry) for entry in wavetables]
    group_names = [name for name, _ in groups]
    assert len(set(group_names)) == len(
        group_names
    ), f"Wavetable group names must be unique, got {group_names}"

    # 2. Extract unique single wavetables to avoid duplicate distance calculations
    unique_wavetables = get_unique_wavetables(wavetables)
    log.info(
        f"Computing distances for {len(unique_wavetables)} unique wavetables (no duplicate computation)"
    )

    # 3. Compute distances on single wavetables and save to TSV
    compute_distances(
        loss_fns=loss_fns,
        wavetables=unique_wavetables,
        mod_sig_references=mod_sig_references,
        samples_dir=samples_dir,
        save_path=tsv_path,
        sr=sr,
        target_lufs=target_lufs,
        use_rand_phase_shift=use_rand_phase_shift,
        max_shift=max_shift,
        shift_seed=shift_seed,
    )
