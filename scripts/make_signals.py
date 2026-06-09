import glob
import logging
import os

import numpy as np
import torch as tr
import torchaudio

from modulations_mod_extraction import make_mod_signal, make_quasi_periodic
from util import create_wavetable_sweep, loudness_normalize

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    wavetable_dir = os.path.join("../data/listening_test")
    save_dir = "../out/"
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
            mod_sig = make_mod_signal(n_samples, sr, freq, shape="cos")
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

        for reg in amp_vals:
            mod_sig = make_mod_signal(
                n_samples, sr, amp_freq, shape="cos", phase=tr.pi / 2
            )
            mod_sig = amp_center_val + (mod_sig - 0.5) * reg
            log.info(
                f"amp={reg:.2f} mod_sig min={mod_sig.min():.4f} max={mod_sig.max():.4f} mean={mod_sig.mean():.4f}"
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
                f"{wt_name}__amp_{amp_freq:.2f}hz_{reg:.2f}_{target_lufs}lufs.wav"
            )
            save_path = os.path.join(save_dir, save_name)
            torchaudio.save(
                save_path,
                tr.tensor(sweep_norm).unsqueeze(0).expand(2, -1).float(),
                sr,
            )
            log.info(f"Saved {save_name} (loudness={loudness:.1f}, gain={gain:.1f}dB)")

        for reg in reg_vals:
            mod_sig = make_mod_signal(n_samples, sr, reg_freq, shape="cos")
            mod_sig, norm_gaps = make_quasi_periodic(
                mod_sig, randomness=reg, seed=reg_seed
            )
            print(f"r={reg:.2f} intervals: {[f'{g:.2f}' for g in norm_gaps]}")
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
