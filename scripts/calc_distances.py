import argparse
import glob
import logging
import os
import re
from typing import Dict, List, Sequence, Tuple, Union

import pandas as pd
import torch as tr
import torchaudio
from auraloss.freq import MultiResolutionSTFTLoss
from auraloss.time import ESRLoss
from torch import Tensor as T
from torch import nn

from losses import (
    Scat1DLoss,
    PANNsEmbeddingLoss,
    ClapEmbeddingLoss,
    MFCCDistance,
    LogMSSLoss,
    JTFSTLoss,
    EncodecEmbeddingLoss,
    VGGishEmbeddingLoss,
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


def get_phase_info(path: str) -> Tuple[int, int, str]:
    """Extract (phase_idx, n_phases, phase_str) from file path.
    Returns (0, 1, '') if no phase tag is found."""
    m = re.search(r"__phase_(\d+)_(\d+)", os.path.basename(path))
    if m:
        p_idx = int(m.group(1))
        n_p = int(m.group(2))
        return p_idx, n_p, f"__phase_{p_idx}_{n_p}"
    return 0, 1, ""


def find_reference_path(
    samples_dir: str,
    wt_name: str,
    mod_sig: str,
    target_lufs: int,
    phase_idx: int,
    n_phases: int,
) -> str:
    """Find the path to the reference audio sample for a given phase."""
    # 1. Exact match with n_phases
    path_exact = os.path.join(
        samples_dir,
        f"{wt_name}__{mod_sig}_{target_lufs}lufs__phase_{phase_idx}_{n_phases}.wav",
    )
    if os.path.exists(path_exact):
        return path_exact

    # 2. Glob with any n_phases
    matches = sorted(
        glob.glob(
            os.path.join(
                samples_dir,
                f"{wt_name}__{mod_sig}_{target_lufs}lufs__phase_{phase_idx}_*.wav",
            )
        )
    )
    if matches:
        return matches[0]

    # 3. Fallback to unphased file if phase_idx == 0
    if phase_idx == 0:
        path_unphased = os.path.join(
            samples_dir, f"{wt_name}__{mod_sig}_{target_lufs}lufs.wav"
        )
        if os.path.exists(path_unphased):
            return path_unphased

    raise FileNotFoundError(
        f"Missing reference audio for wt='{wt_name}', ref='{mod_sig}', phase={phase_idx} in {samples_dir}"
    )


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
    ref_match_phase: bool = False,
) -> pd.DataFrame:
    """Compute distances for single wavetables and save the result to a TSV file.

    Parameters
    ----------
    loss_fns : Sequence[Union[nn.Module, Tuple[str, nn.Module]]]
        Loss function(s) to evaluate.
    wavetables : Sequence[str]
        List of wavetable names to evaluate.
    mod_sig_references : Sequence[str]
        Reference modulation signals (e.g. 'amp_1.00hz_0.10', 'freq_0.25hz', 'reg_1.00hz_0.000').
    samples_dir : str
        Directory containing synthesized audio samples.
    save_path : str
        Path to output TSV file.
    sr : int, default=44100
        Sample rate.
    target_lufs : int, default=-18
        Loudness normalization level used in filenames.
    use_rand_phase_shift : bool, default=False
        Whether to apply random circular shift to reference audio.
    max_shift : int, default=2048
        Maximum shift in samples when use_rand_phase_shift is True.
    shift_seed : int, default=42
        Random seed for shift generator.
    ref_match_phase : bool, default=False
        If True, compare each variant at phase_n against the reference audio with
        the corresponding phase_n.
        If False, always compare each variant against the reference audio at phase 0.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    suffix = f"_{target_lufs}lufs.wav"
    rand_gen = tr.Generator().manual_seed(shift_seed)

    loss_entries = [resolve_loss_fn(entry) for entry in loss_fns]
    loss_names = [name for name, _ in loss_entries]
    assert len(set(loss_names)) == len(
        loss_names
    ), f"Loss function names must be unique, got {loss_names}"

    ref_audio_cache: Dict[str, T] = {}
    rows = []

    for loss_name, loss_fn in loss_entries:
        for wt_name in wavetables:
            for mod_sig in mod_sig_references:
                _, ref_amount, _ = parse_amount(mod_sig)
                variant_paths = find_variants(samples_dir, wt_name, mod_sig, suffix)
                log.info(
                    f"{loss_name} | {wt_name} | {mod_sig}: found {len(variant_paths)} samples"
                )

                for variant_path in variant_paths:
                    variant_name = os.path.basename(variant_path)
                    if variant_name.endswith(".wav"):
                        variant_name = variant_name[:-4]

                    phase_idx, n_phases, phase_str = get_phase_info(variant_path)

                    # Extract stimulus tag without wt prefix, phase suffix, or lufs
                    core = variant_name[len(f"{wt_name}__") :]
                    if phase_str and core.endswith(phase_str):
                        core = core[: -len(phase_str)]
                    lufs_tag = f"_{target_lufs}lufs"
                    if core.endswith(lufs_tag):
                        stim_tag = core[: -len(lufs_tag)]
                    else:
                        stim_tag = re.sub(r"_[+-]?\d+lufs$", "", core)

                    _, amount, _ = parse_amount(stim_tag)
                    variant_mod_sig = f"{stim_tag}{phase_str}"

                    # Determine reference phase based on ref_match_phase boolean option
                    target_ref_phase = phase_idx if ref_match_phase else 0
                    ref_path = find_reference_path(
                        samples_dir=samples_dir,
                        wt_name=wt_name,
                        mod_sig=mod_sig,
                        target_lufs=target_lufs,
                        phase_idx=target_ref_phase,
                        n_phases=n_phases,
                    )

                    if ref_path not in ref_audio_cache:
                        ref_audio_cache[ref_path] = load_audio(ref_path, sr)
                    ref_audio = ref_audio_cache[ref_path]

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

                    is_reference = (amount == ref_amount) and (
                        phase_idx == target_ref_phase
                    )

                    rows.append(
                        {
                            "loss_fn": loss_name,
                            "wavetable": wt_name,
                            "mod_type": mod_sig.split("_", 1)[0],
                            "reference": mod_sig,
                            "ref_amount": ref_amount,
                            "mod_sig": variant_mod_sig,
                            "amount": amount,
                            "phase": phase_idx,
                            "ref_phase": target_ref_phase,
                            "is_reference": is_reference,
                            "ref_shift": shift,
                            "distance": dist,
                        }
                    )
                    log.info(
                        f"  {variant_mod_sig} vs ref_phase_{target_ref_phase}: "
                        f"{dist:.6g} (shift={shift})"
                    )

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, sep="\t")
    log.info(f"Saved {len(df)} distances to {save_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute audio loss distances across stimulus variants and phases."
    )
    parser.add_argument(
        "--samples-dir",
        # default="../out/audio_samples",
        default="../out/audio_samples_28_phases",
        help="Directory containing audio samples (default: ../out/audio_samples)",
    )
    parser.add_argument(
        "--save-dir",
        default="../out/distances",
        help="Directory to save distance TSV (default: ../out/distances)",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Explicit TSV path (default: {save-dir}/distances_testing.tsv)",
    )
    parser.add_argument(
        "--ref-match-phase",
        action="store_true",
        default=False,
        # default=True,
        help=(
            "If True, compare each variant at phase_n against reference at phase_n. "
            "If False (default), always compare against reference at phase 0."
        ),
    )
    args, unknown = parser.parse_known_args()

    samples_dir = args.samples_dir
    save_dir = args.save_dir
    # tsv_path = args.save_path or os.path.join(save_dir, "distances_phases_mss_log_lin.tsv")
    # tsv_path = args.save_path or os.path.join(save_dir, "distances_phases_matched_mss_log_lin.tsv")
    # tsv_path = args.save_path or os.path.join(save_dir, "distances_phases_matched_actually_mss_rev.tsv")
    # tsv_path = args.save_path or os.path.join(save_dir, "distances_phases_mfcc.tsv")
    # tsv_path = args.save_path or os.path.join(save_dir, "distances_phases_matched_mfcc.tsv")
    # tsv_path = args.save_path or os.path.join(save_dir, "distances_phases_vggish.tsv")
    # tsv_path = args.save_path or os.path.join(save_dir, "distances_phases_matched_vggish.tsv")
    # tsv_path = args.save_path or os.path.join(save_dir, "distances_phases_clap2.tsv")
    # tsv_path = args.save_path or os.path.join(save_dir, "distances_phases_encodec48.tsv")
    # tsv_path = args.save_path or os.path.join(save_dir, "distances_phases_panns_wavegram_logmel.tsv")
    tsv_path = args.save_path or os.path.join(save_dir, "distances_phases_scat1d.tsv")
    ref_match_phase = args.ref_match_phase
    sr = 44100
    target_lufs = -18
    use_rand_phase_shift = False
    max_shift = 2048  # Two wavetable frames (44100 / 1024 Hz carrier)
    shift_seed = 42

    loss_fns = [
        # ("mse", nn.MSELoss()),
        # ("esr", ESRLoss()),
        # ("mss", MultiResolutionSTFTLoss()),
        # ("mss_log_lin", MultiResolutionSTFTLoss(
        #     fft_sizes=[64, 128, 256, 512, 1024, 2048],
        #     hop_sizes=[16, 32, 64, 128, 256, 512],
        #     win_lengths=[64, 128, 256, 512, 1024, 2048],
        #     w_sc=0.0,
        #     w_phs=0.0,
        #     w_lin_mag=1.0,
        #     w_log_mag=1.0,
        #     mag_distance="L1",
        # )),
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
        # ("mfcc_p2", MFCCDistance(sr=sr, p=2)),
        # ("vggish", VGGishEmbeddingLoss(in_sr=sr)),
        # ("vggish_tv", VGGishEmbeddingLoss(in_sr=sr, use_time_varying=True)),
        # ("clap2", ClapEmbeddingLoss(use_cuda=False, in_sr=sr)),
        # ("encodec48", EncodecEmbeddingLoss(in_sr=sr)),
        # ("encodec48_tv", EncodecEmbeddingLoss(in_sr=sr, use_time_varying=True)),
        # ("panns_cnn14_32k", PANNsEmbeddingLoss(variant="cnn14-32k", in_sr=sr)),
        # (
        #     "panns_wavegram_logmel",
        #     PANNsEmbeddingLoss(variant="wavegram-logmel", in_sr=sr),
        # ),
        # ("scat1d", Scat1DLoss(shape=176400, J=12, Q1=8, Q2=2, T=None, max_order=2, p=2)),
        ("scat1d_log1p", Scat1DLoss(shape=176400, J=12, Q1=8, Q2=2, T=None, max_order=2, p=2, use_rho_log1p=True)),
        # ("scat1d_cqt", Scat1DLoss(shape=176400, J=12, Q1=8, Q2=2, T=1, max_order=1, p=2)),
        # ("jtfs", JTFSTLoss(shape=176400, J=12, Q1=8, Q2=2, J_fr=3, Q_fr=2, T=None, F=None, format_="joint", p=2)),
        # ("jtfs2", JTFSTLoss(shape=176400, J=12, Q1=8, Q2=2, J_fr=5, Q_fr=2, T=2048, F=1, format_="joint", p=2, use_rho_log1p=True)),
        # ("jtfs_log1p", JTFSTLoss(shape=176400, J=12, Q1=8, Q2=2, J_fr=5, Q_fr=2, T=None, F=None, format_="joint", p=2, use_rho_log1p=True)),
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
        ref_match_phase=ref_match_phase,
    )
