"""Do the neural audio embeddings encode human perceptual distance?

This script takes the exact stimulus pairs that ``calc_distances.py`` measures and
asks how much of the *listener* distance each embedding accounts for. It is
deliberately structured around three progressively weaker claims, because they are
easy to conflate and only the first one is answered by a distance function alone:

  1. ZERO-SHOT  -- Does the model's off-the-shelf distance (L2 / cosine on the
                   embedding) already rank pairs the way listeners do? This is the
                   claim that matters if you want to *use* the embedding as a loss.
  2. PROBE      -- Is the information linearly decodable from the embedding
                   geometry even when plain L2 fails? We fit a diagonal Mahalanobis
                   metric (a non-negative re-weighting of embedding dimensions) and
                   require it to generalise to a wavetable it never saw. This is the
                   claim "the information is in there, the default metric just
                   doesn't expose it".
  3. ORDINAL    -- On the comparisons where listeners agree with *each other*, how
                   often does the model agree with listeners? Interpretable, and
                   robust to the arbitrary units of a rating scale.

Claim 2 is unfalsifiable without a capacity constraint (a big enough MLP fits 90
points of pure noise), so the operational version here is "linearly decodable AND
generalises across wavetables", plus a permutation null that re-runs the entire
fitting pipeline on shuffled targets.

Everything is reported relative to a NOISE CEILING estimated by split-half
reliability of the listeners themselves. A model at rho=0.55 against a ceiling of
0.60 is essentially perfect; the same number against a ceiling of 0.95 is weak.

WHAT IS SIMULATED
-----------------
The real listening test data is not in this repo, so ``simulate_human_ratings``
fabricates it: 50 participants, one 0-100 rating per participant per pair, with
per-participant gain/bias, trial noise and occasional lapses. It is written to
``out/neural_analysis/human_ratings_placeholder.csv``. Swap in the real data by
pointing HUMAN_RATINGS_CSV at a file with the same tidy columns
(``participant, pair_id, rating``) -- nothing else in the script changes.

Because the placeholder ratings are generated *from the synthesis parameters*, the
"ground truth parameter" baseline is at ceiling by construction and the embeddings
can only look good to the extent that they track those parameters. Treat the
numbers as a smoke test of the pipeline, not as a result.

USAGE
-----
    python scripts/neural_analysis.py

Embeddings are cached to ``out/embeddings/<model>/<stimulus>.npz`` so that the
statistics can be iterated on without re-running the networks. Delete that
directory to force re-extraction.

Expect roughly 10 minutes on CPU after the checkpoints are downloaded. Almost all of
it is the permutation nulls and bootstraps (every permutation re-fits the probe);
drop N_PERMUTATIONS and N_BOOTSTRAPS to 100 while iterating.
"""

import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as tr
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
# The model and loss code lives in code/, which is a source root in the IDE but not
# on sys.path for a plain `python scripts/...` invocation.
for _p in (SCRIPT_DIR, os.path.join(REPO_DIR, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the pair enumeration that calc_distances.py uses rather than re-deriving it,
# so the two scripts cannot drift apart on which pairs are being compared. These live in
# util.py rather than calc_distances.py so that this script does not have to import the
# loss functions (and therefore CLAP, kymatio and auraloss) just to parse a filename.
from util import find_variants, parse_amount

# Matches the plot styling in calc_distances.py.
SERIES_COLOR = "#2a78d6"
AXIS_COLOR = "#52514e"

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


# ======================================================================================
# Configuration
# ======================================================================================

SAMPLES_DIR = os.path.join(REPO_DIR, "out", "samples_all")
EMB_DIR = os.path.join(REPO_DIR, "out", "embeddings")
SAVE_DIR = os.path.join(REPO_DIR, "out", "neural_analysis")

SR = 44100
TARGET_LUFS = -18
SUFFIX = f"_{TARGET_LUFS}lufs.wav"

# Same wavetables and reference modulations as calc_distances.py.
WAVETABLES = [
    "brightness_real__harmonics__synced_sines__256_1024",
    "brightness_synthetic__256_1024",
    "richness_real__filter__acid_saw__46_1024__inverted",
    "richness_synthetic__256_1024",
    "warmth_real__vintage__logue_saw__166_1024",
    "warmth_synthetic__256_1024",
]
MOD_SIG_REFERENCES = [
    "amp_1.00hz_0.10",
    "freq_0.25hz",
    "reg_1.00hz_0.000",
]
# The fastest modulation rate present in the stimuli. Used to flag representations
# whose frame rate is too low to represent the manipulation at all.
MAX_STIMULUS_MOD_RATE_HZ = 4.0

# calc_distances.py compares each reference against every variant *including itself*,
# so each (wavetable, mod_type) block contributes 5 pairs, one of which is trivial
# (distance 0). Keeping the self pair anchors the low end of the rating scale, but it is
# also the easiest pair in the set, so it inflates every correlation somewhat. Flip this
# to True and re-run to check that no conclusion depends on it.
DROP_SELF_PAIRS = False

# Placeholder listening test.
N_PARTICIPANTS = 50
RATING_MIN, RATING_MAX = 0.0, 100.0
HUMAN_SEED = 0
# Point this at real data (columns: participant, pair_id, rating) to use it instead.
HUMAN_RATINGS_CSV: Optional[str] = None

# Modulation spectrum readout: the band we keep after taking an FFT along the time
# axis of the embedding trajectory. The stimuli modulate at 0.25-4 Hz.
MOD_SPEC_FMIN_HZ = 0.1
MOD_SPEC_FMAX_HZ = 20.0

# Probe.
# Must bracket the cross-validated optimum at BOTH ends or the probe is limited by the
# grid rather than by the features; PrefactorizedRidgeCV warns when it does not. 20
# decades at half-decade spacing: the upper end matters because holding out a whole
# modulation type often makes "shrink to the mean" the best available fit, and the lower
# end matters for small well-conditioned feature sets like the parameter baseline.
RIDGE_ALPHAS = np.logspace(-8, 12, 41)
# These two dominate the runtime (every permutation re-runs the whole probe). 500 gives
# a p-value resolution of ~0.002, which is plenty; raise them for final numbers.
N_PERMUTATIONS = 500
N_BOOTSTRAPS = 500
PCA_MATCH_DIMS = 32  # Capacity-matched control: every model reduced to this many dims
# Below this, dividing a model correlation by the ceiling amplifies noise more than it
# corrects for it, so rho_over_ceiling is reported as NaN instead.
MIN_USABLE_CEILING = 0.2
ANALYSIS_SEED = 42

DPI = 150


# ======================================================================================
# Part 1 -- Stimuli and pairs
# ======================================================================================


def build_pairs() -> pd.DataFrame:
    """Enumerate exactly the pairs that calc_distances.py measures.

    For each wavetable and each reference modulation, calc_distances.py finds every
    variant that differs only in its amount and measures reference-vs-variant. That
    gives 6 wavetables x 3 mod types x 5 levels = 90 pairs, of which 18 are the
    trivial self comparison.

    A "block" is one (wavetable, mod_type) cell. Blocks are the unit of analysis
    throughout: correlations are computed within block and cross-validation holds out
    whole wavetables, because pooling across blocks lets a trivial "amp pairs are all
    bigger than freq pairs" main effect masquerade as perceptual agreement.
    """
    rows = []
    for wt_name in WAVETABLES:
        for ref_mod_sig in MOD_SIG_REFERENCES:
            ref_stim = f"{wt_name}__{ref_mod_sig}"
            ref_path = os.path.join(SAMPLES_DIR, f"{ref_stim}{SUFFIX}")
            assert os.path.exists(ref_path), f"Missing reference {ref_path}"
            _, ref_amount, _ = parse_amount(ref_mod_sig)
            mod_type = ref_mod_sig.split("_", 1)[0]

            for var_path in find_variants(SAMPLES_DIR, wt_name, ref_mod_sig, SUFFIX):
                var_stim = os.path.basename(var_path)[: -len(SUFFIX)]
                var_mod_sig = var_stim[len(f"{wt_name}__") :]
                _, var_amount, _ = parse_amount(var_mod_sig)
                rows.append(
                    {
                        "pair_id": f"{ref_stim}__VS__{var_mod_sig}",
                        "block": f"{wt_name}__{mod_type}",
                        "wavetable": wt_name,
                        "mod_type": mod_type,
                        "ref_stim": ref_stim,
                        "var_stim": var_stim,
                        "ref_amount": ref_amount,
                        "var_amount": var_amount,
                        "is_self_pair": var_amount == ref_amount,
                    }
                )
    df = pd.DataFrame(rows)
    assert df["pair_id"].is_unique, "Pair ids must be unique"
    if DROP_SELF_PAIRS:
        df = df[~df["is_self_pair"]]
    df = df.reset_index(drop=True)  # Positional indexing is assumed everywhere below
    log.info(
        f"Built {len(df)} pairs over {df['block'].nunique()} blocks "
        f"({df['is_self_pair'].sum()} self pairs)"
    )
    return df


def stimulus_names(pairs: pd.DataFrame) -> List[str]:
    """Every distinct audio file referenced by the pair table."""
    return sorted(set(pairs["ref_stim"]) | set(pairs["var_stim"]))


def stimulus_path(stim: str) -> str:
    return os.path.join(SAMPLES_DIR, f"{stim}{SUFFIX}")


# ======================================================================================
# Part 2 -- Human ratings (PLACEHOLDER)
# ======================================================================================

# Monotone maps from a synthesis parameter to a putative perceptual coordinate. These
# only exist to make the simulated data look like plausible listening test data; they
# are never used by the analysis itself (that would leak the answer into the probe).
_PERCEPTUAL_WARP = {
    "amp": lambda a: np.power(a, 0.8),  # Modulation depth, mildly compressive
    "freq": lambda a: np.log2(a),  # Rate, perceived roughly in octaves
    "reg": lambda a: np.sqrt(a),  # Irregularity, saturates quickly
}
_RESPONSE_COMPRESSION = 2.0  # Larger -> listeners compress big differences more


def _latent_distance(pairs: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """The 'true' perceptual distance the simulated listeners are noisy reads of.

    Built as |warp(a) - warp(b)| normalised per mod type to [0, 1], scaled by a
    per-wavetable salience gain (some timbres make modulation more obvious), then
    pushed through a saturating response so that the top of the 0-100 scale is
    compressed the way rating scales usually are.
    """
    # Per-wavetable salience, drawn once and shared by all participants.
    gains = {wt: g for wt, g in zip(WAVETABLES, rng.uniform(0.75, 1.25, len(WAVETABLES)))}

    latent = np.zeros(len(pairs))
    mod_types = pairs["mod_type"].to_numpy()
    ref_amt = pairs["ref_amount"].to_numpy()
    var_amt = pairs["var_amount"].to_numpy()
    for mod_type in np.unique(mod_types):
        mask = mod_types == mod_type
        warp = _PERCEPTUAL_WARP[mod_type]
        a, b = warp(ref_amt[mask]), warp(var_amt[mask])
        span = max(np.ptp(np.concatenate([a, b])), 1e-12)
        latent[mask] = np.abs(a - b) / span

    gain = pairs["wavetable"].map(gains).to_numpy()
    x = np.clip(gain * latent, 0.0, None)
    # Saturating response normalised so that latent == 1 maps to the top of the scale.
    denom = 1.0 - np.exp(-_RESPONSE_COMPRESSION)
    return RATING_MAX * (1.0 - np.exp(-_RESPONSE_COMPRESSION * x)) / denom


def simulate_human_ratings(pairs: pd.DataFrame, seed: int = HUMAN_SEED) -> pd.DataFrame:
    """Fabricate a listening test: N_PARTICIPANTS ratings in [0, 100] per pair.

    The noise model is what makes this useful as a placeholder -- a clean latent
    distance would give a noise ceiling of 1.0 and hide the fact that every reported
    correlation is bounded by listener reliability. Three sources of noise:

      * per-participant gain and bias  -- people use the scale differently
      * per-trial gaussian noise       -- within-participant inconsistency
      * lapses                         -- occasional uniform-random responses

    Returns tidy long form: one row per (participant, pair).
    """
    rng = np.random.default_rng(seed)
    d_true = _latent_distance(pairs, rng)

    gain = rng.lognormal(mean=0.0, sigma=0.18, size=N_PARTICIPANTS)
    bias = rng.normal(loc=0.0, scale=6.0, size=N_PARTICIPANTS)
    trial_noise = rng.normal(loc=0.0, scale=18.0, size=(N_PARTICIPANTS, len(pairs)))

    ratings = gain[:, None] * d_true[None, :] + bias[:, None] + trial_noise

    # ~8% of trials are inattentive and effectively random.
    lapse = rng.random((N_PARTICIPANTS, len(pairs))) < 0.08
    ratings[lapse] = rng.uniform(RATING_MIN, RATING_MAX, lapse.sum())
    ratings = np.round(np.clip(ratings, RATING_MIN, RATING_MAX))

    long = pd.DataFrame(
        {
            "participant": np.repeat(np.arange(N_PARTICIPANTS), len(pairs)),
            "pair_id": np.tile(pairs["pair_id"].to_numpy(), N_PARTICIPANTS),
            "rating": ratings.reshape(-1),
        }
    )
    log.warning(
        f"Using SIMULATED listening test data ({N_PARTICIPANTS} participants, "
        f"{len(long)} ratings). Set HUMAN_RATINGS_CSV to use real data."
    )
    return long


def load_human_ratings(path: str, pairs: pd.DataFrame) -> pd.DataFrame:
    """Load real ratings and check they cover exactly the pairs we are analysing."""
    long = pd.read_csv(path)
    missing = {"participant", "pair_id", "rating"} - set(long.columns)
    assert not missing, f"{path} is missing columns {missing}"
    long = long[long["pair_id"].isin(set(pairs["pair_id"]))]
    covered = set(long["pair_id"])
    absent = set(pairs["pair_id"]) - covered
    assert not absent, f"No ratings for {len(absent)} pairs, e.g. {sorted(absent)[:3]}"
    log.info(f"Loaded {len(long)} ratings from {path}")
    return long


def ratings_to_matrix(long: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """Pivot tidy ratings into a dense (n_participants, n_pairs) matrix.

    The matrix form is what the bootstrap and the split-half noise ceiling need: both
    resample *participants*, i.e. rows.
    """
    wide = long.pivot_table(
        index="participant", columns="pair_id", values="rating", aggfunc="mean"
    )
    wide = wide.reindex(columns=pairs["pair_id"].to_numpy())
    assert not wide.isna().any().any(), "Ratings matrix has holes"
    return wide.to_numpy(dtype=np.float64)


# ======================================================================================
# Part 3 -- Embedding extraction and caching
# ======================================================================================
#
# Two things matter here beyond "run the model".
#
# 1. TIME RESOLUTION. code/losses.py:177 mean-pools the embedding over frames before
#    taking a distance. The stimuli modulate at 0.25-4 Hz over a 4 s window, i.e. 1 to
#    16 cycles, so averaging over time is close to a matched filter *against* the thing
#    that varies across the freq levels. A null result for a time-averaged embedding is
#    a result about the pooling, not about the model. So every representation is kept
#    frame-wise where the architecture allows it, and each one records its frame rate so
#    we can flag layers whose Nyquist rate is below the fastest stimulus modulation.
#
# 2. DEPTH. PANNs discards its time axis at the very end (max+mean pool -> fc1), so the
#    2048-d clip embedding that calc_distances.py uses cannot represent modulation rate
#    even in principle. Forward hooks on the six conv blocks give the same network's
#    intermediate representations, which do have a time axis.


class Repr:
    """One representation of one stimulus: (n_frames, n_features) plus its frame rate.

    Clip-level representations are stored as n_frames == 1 with fps == 0.0, which makes
    the "this readout is impossible here" cases explicit rather than silently wrong.
    """

    def __init__(self, array: np.ndarray, fps: float):
        assert array.ndim == 2, f"Expected (n_frames, n_features), got {array.shape}"
        self.array = np.ascontiguousarray(array, dtype=np.float32)
        self.fps = float(fps)

    @property
    def is_clip_level(self) -> bool:
        return self.array.shape[0] == 1

    @property
    def nyquist_hz(self) -> float:
        return self.fps / 2.0


class EmbeddingExtractor(ABC):
    """Turns one mono waveform into a dict of named representations.

    Subclasses only deal with the model; resampling, batching, caching and the readouts
    are handled by the driver so that every model is treated identically.
    """

    #: Whether a raw (unstandardised) L2 in these units is a meaningful distance. False
    #: for hand-crafted features whose dimensions are in incomparable units (Hz vs dB).
    native_metric = True

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def model_sr(self) -> int:
        pass

    @abstractmethod
    def representations(self, audio: tr.Tensor) -> Dict[str, Repr]:
        """audio is (1, n_samples) mono at self.model_sr()."""
        pass


def _randomise_weights(model: tr.nn.Module, seed: int) -> None:
    """Re-initialise a network with the same scheme PANNs uses, for the random control.

    This is the single most important control in the script. If a randomly initialised
    network probes as well as the trained one, the finding is "a random projection of a
    mel spectrogram carries modulation information", not "AudioSet training taught the
    model about perceptual distance". BatchNorm running statistics are reset too,
    otherwise the trained normalisation would keep doing useful work under random
    weights. The STFT/mel front end is left alone -- it is fixed, not learned, and the
    control is about the learned part.
    """
    tr.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, (tr.nn.Conv1d, tr.nn.Conv2d, tr.nn.Linear)):
            tr.nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                m.bias.data.fill_(0.0)
        elif isinstance(m, (tr.nn.BatchNorm1d, tr.nn.BatchNorm2d)):
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)
            if m.running_mean is not None:
                m.running_mean.zero_()
                m.running_var.fill_(1.0)


class PANNsExtractor(EmbeddingExtractor):
    """PANNs CNN14 / Wavegram-Logmel, with per-conv-block frame-wise representations.

    Returns:
      conv_block1..6 -- (n_frames, n_channels), averaged over the mel axis. Frame rate
                        halves at each block, so conv_block1 runs at ~50 fps and
                        conv_block6 at ~3 fps.
      clip           -- (1, 2048), the embedding calc_distances.py actually uses.
    """

    def __init__(self, variant: str, randomise: bool = False, seed: int = 0):
        suffix = "__random" if randomise else ""
        super().__init__(f"panns_{variant.replace('-', '_')}{suffix}")
        from panns.model_loader import PANNsModel

        self.variant = variant
        self.sr = 16000 if variant == "cnn14-16k" else 32000
        self.panns = PANNsModel(variant=variant)
        self.panns.load_model()
        if randomise:
            _randomise_weights(self.panns.model, seed)
        self.panns.eval()
        for p in self.panns.parameters():
            p.requires_grad = False

        # Discover conv blocks by name instead of hardcoding, so both variants work.
        self._acts: Dict[str, tr.Tensor] = {}
        self._block_names = []
        for mod_name, module in self.panns.model.named_children():
            if mod_name.startswith("conv_block"):
                self._block_names.append(mod_name)
                module.register_forward_hook(self._make_hook(mod_name))
        self._block_names.sort(key=lambda n: int(n.replace("conv_block", "")))
        log.info(f"{self.name}: hooked {self._block_names}")

    def _make_hook(self, mod_name: str):
        def hook(_module, _inputs, output):
            self._acts[mod_name] = output.detach()

        return hook

    def model_sr(self) -> int:
        return self.sr

    def representations(self, audio: tr.Tensor) -> Dict[str, Repr]:
        self._acts.clear()
        clip = self.panns.get_embedding(audio)  # (1, 2048), fires the hooks
        dur_sec = audio.size(-1) / self.sr

        reprs: Dict[str, Repr] = {}
        for mod_name in self._block_names:
            act = self._acts[mod_name]  # (1, n_channels, n_frames, n_mels)
            assert act.ndim == 4, f"Unexpected hook output {act.shape}"
            # Average over the mel axis: we care about how each channel's activation
            # fluctuates in time, not about its spectral profile within a frame.
            act = act.mean(dim=-1).squeeze(0).transpose(0, 1)  # (n_frames, n_channels)
            arr = act.numpy()
            reprs[mod_name] = Repr(arr, fps=arr.shape[0] / dur_sec)

        clip = clip.reshape(1, -1).numpy()
        reprs["clip"] = Repr(clip, fps=0.0)
        return reprs


class ClapExtractor(EmbeddingExtractor):
    """Microsoft CLAP audio encoder, matching how ClapEmbeddingLoss calls it.

    CLAP expects a fixed-length input, so audio is repeated/truncated exactly the way
    EmbeddingLoss.preproc_audio does it. The encoder returns a clip-level vector, so
    the frame-wise readouts are unavailable here -- which is itself worth reporting.
    """

    def __init__(self, version: str = "2023", use_cuda: bool = False):
        super().__init__("clap")
        from msclap import CLAP

        self.clap = CLAP(version=version, use_cuda=use_cuda)
        self.sr = self.clap.args.sampling_rate
        self.n_samples = int(self.clap.args.duration * self.sr)

    def model_sr(self) -> int:
        return self.sr

    def representations(self, audio: tr.Tensor) -> Dict[str, Repr]:
        n = audio.size(-1)
        if n < self.n_samples:
            audio = audio.repeat(1, self.n_samples // n + 1)
        audio = audio[:, : self.n_samples]
        with tr.no_grad():
            emb, _ = self.clap.clap.audio_encoder(audio)
        emb = emb.reshape(1, -1).numpy()
        return {"clip": Repr(emb, fps=0.0)}


class LowLevelFeatureExtractor(EmbeddingExtractor):
    """Frame-wise signal descriptors, as the "does the embedding beat DSP 101" baseline.

    Five features per frame at ~50 fps: log RMS, spectral centroid, spectral spread,
    spectral flatness and 85% rolloff. Slotting these into the same extractor interface
    means they go through identical readouts, probing and cross-validation, so the
    comparison against the neural models is apples to apples.
    """

    native_metric = False  # Dimensions are in Hz, dB and unitless -- raw L2 is nonsense

    def __init__(self, n_fft: int = 2048, hop: int = 882):
        super().__init__("lowlevel")
        self.n_fft = n_fft
        self.hop = hop  # 882 @ 44.1 kHz -> 50 fps, Nyquist 25 Hz >> 4 Hz mod rate
        self.window = tr.hann_window(n_fft)

    def model_sr(self) -> int:
        return SR

    def representations(self, audio: tr.Tensor) -> Dict[str, Repr]:
        spec = tr.stft(
            audio.squeeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        ).abs()  # (n_bins, n_frames)
        power = spec**2 + 1e-12
        freqs = tr.linspace(0.0, SR / 2, spec.size(0)).unsqueeze(1)

        total = power.sum(dim=0)
        centroid = (freqs * power).sum(dim=0) / total
        spread = tr.sqrt(((freqs - centroid) ** 2 * power).sum(dim=0) / total)
        flatness = tr.exp(tr.log(power).mean(dim=0)) / (total / power.size(0))
        cumulative = tr.cumsum(power, dim=0) / total.unsqueeze(0)
        rolloff = freqs.squeeze(1)[(cumulative < 0.85).sum(dim=0).clamp(max=spec.size(0) - 1)]
        log_rms = 10.0 * tr.log10(total + 1e-12)

        feats = tr.stack([log_rms, centroid, spread, flatness, rolloff], dim=1)
        arr = feats.numpy()
        dur_sec = audio.size(-1) / SR
        return {"descriptors": Repr(arr, fps=arr.shape[0] / dur_sec)}


def _load_mono(path: str, target_sr: int) -> tr.Tensor:
    """Load a stimulus as (1, n_samples) mono at target_sr.

    The .wav files are mono duplicated across two channels, same as calc_distances.py.
    """
    import torchaudio

    audio, sr = torchaudio.load(path)
    assert sr == SR, f"Expected sr={SR}, got {sr} for {path}"
    audio = audio[:1, :]
    if target_sr != SR:
        audio = torchaudio.transforms.Resample(orig_freq=SR, new_freq=target_sr)(audio)
    return audio


def extract_all(
    extractor: EmbeddingExtractor, stims: List[str], cache_dir: str = EMB_DIR
) -> Dict[str, Dict[str, Repr]]:
    """Run one model over every stimulus once, caching to disk.

    calc_distances.py re-encodes both sides of every pair, so a stimulus appearing in
    k pairs is encoded k times. Here each of the ~105 stimuli is encoded exactly once
    and every downstream analysis is pure numpy, which is what makes the 1000-fold
    permutation test and the bootstraps affordable.
    """
    # NOTE: the cache is keyed on extractor.name alone. If you change an extractor's
    # configuration without changing its name (e.g. LowLevelFeatureExtractor's n_fft),
    # delete the corresponding directory or you will silently reuse the old features.
    out_dir = os.path.join(cache_dir, extractor.name)
    os.makedirs(out_dir, exist_ok=True)
    result: Dict[str, Dict[str, Repr]] = {}

    for i, stim in enumerate(stims):
        cache_path = os.path.join(out_dir, f"{stim}.npz")
        if os.path.exists(cache_path):
            with np.load(cache_path) as data:
                keys = [k for k in data.files if not k.endswith("__fps")]
                result[stim] = {
                    k: Repr(data[k], float(data[f"{k}__fps"])) for k in keys
                }
            continue

        audio = _load_mono(stimulus_path(stim), extractor.model_sr())
        with tr.no_grad():
            reprs = extractor.representations(audio)
        payload = {}
        for k, r in reprs.items():
            payload[k] = r.array
            payload[f"{k}__fps"] = np.asarray(r.fps)
        np.savez_compressed(cache_path, **payload)
        result[stim] = reprs
        if (i + 1) % 20 == 0:
            log.info(f"{extractor.name}: encoded {i + 1}/{len(stims)}")

    any_reprs = next(iter(result.values()))
    for k, r in any_reprs.items():
        covers = r.nyquist_hz >= MAX_STIMULUS_MOD_RATE_HZ
        log.info(
            f"{extractor.name}.{k}: {r.array.shape} @ {r.fps:.2f} fps "
            f"(Nyquist {r.nyquist_hz:.2f} Hz, covers {MAX_STIMULUS_MOD_RATE_HZ} Hz "
            f"modulation: {covers})"
        )
    return result


# ======================================================================================
# Part 4 -- Readouts and pair features
# ======================================================================================
#
# A readout turns the per-stimulus representation into the thing we actually compare.
# Three of them, in increasing appropriateness for a modulation experiment:
#
#   time_avg  -- mean over frames. Reproduces code/losses.py:177 exactly, so it is the
#                baseline that says what the current loss functions can see.
#   frames    -- keep the trajectory; the pair feature is the per-channel mean squared
#                frame difference, i.e. a time-invariant diagonal metric.
#   mod_spec  -- FFT along the time axis of each channel's trajectory, keep the
#                0.1-20 Hz magnitude bins. This is targeted directly at the stimulus
#                manipulation: an amplitude modulation at 2 Hz shows up as energy in
#                the 2 Hz bin regardless of its phase. If the information is anywhere,
#                it is most exposed here.
#
# In all three cases the pair feature is an elementwise SQUARED DIFFERENCE, so a linear
# model on top of it is exactly a diagonal Mahalanobis metric: d(a,b) = sum_i w_i
# (e_a[i] - e_b[i])^2. That is the capacity constraint that makes claim 2 falsifiable.

READOUTS = ["time_avg", "frames", "mod_spec"]


def readout_array(r: Repr, readout: str) -> Optional[np.ndarray]:
    """Apply a readout to one stimulus. Returns (n_rows, n_cols) or None if impossible.

    None means "this model cannot support this readout" (e.g. a clip-level embedding has
    no time axis to take a modulation spectrum of), which is reported rather than
    silently skipped -- for PANNs' clip embedding it is arguably the main finding.
    """
    arr = r.array
    if readout == "time_avg":
        return arr.mean(axis=0, keepdims=True)
    if readout == "frames":
        return arr
    if readout == "mod_spec":
        n_frames = arr.shape[0]
        if n_frames < 4:
            return None
        # Remove the DC component first: the modulation spectrum should describe how the
        # channel fluctuates, not how loud it is on average (that is what time_avg is).
        centred = arr - arr.mean(axis=0, keepdims=True)
        window = np.hanning(n_frames)[:, None]  # Limits leakage at 1 cycle per window
        spec = np.abs(np.fft.rfft(centred * window, axis=0)) / n_frames
        freqs = np.fft.rfftfreq(n_frames, d=1.0 / r.fps)
        fmax = min(MOD_SPEC_FMAX_HZ, r.nyquist_hz)
        keep = (freqs >= MOD_SPEC_FMIN_HZ) & (freqs <= fmax)
        if keep.sum() == 0:
            return None
        return spec[keep]
    raise ValueError(f"Unknown readout {readout}")


class PairFeatures:
    """Everything one (model, layer, readout) contributes, evaluated on the pair set.

    Z            -- (n_pairs, n_features) squared differences, the probe's input
    zero_shot    -- named off-the-shelf distances that involve no fitting at all
    covers_rate  -- whether this representation's frame rate satisfies Nyquist for the
                    fastest stimulus modulation. False means a null result here is
                    trivially explained by temporal downsampling rather than by the
                    representation. None means the question does not apply, either
                    because the readout discards the time axis (time_avg) or because
                    there is no time axis to begin with.
    """

    def __init__(
        self,
        model: str,
        layer: str,
        readout: str,
        Z: np.ndarray,
        zero_shot: Dict[str, np.ndarray],
        fps: float,
        n_dims: int,
    ):
        self.model = model
        self.layer = layer
        self.readout = readout
        self.Z = Z
        self.zero_shot = zero_shot
        self.fps = fps
        self.n_dims = n_dims
        if readout == "time_avg" or fps <= 0.0:
            self.covers_rate = None
        else:
            self.covers_rate = fps / 2.0 >= MAX_STIMULUS_MOD_RATE_HZ

    @property
    def key(self) -> str:
        return f"{self.model}|{self.layer}|{self.readout}"


def build_pair_features(
    model_name: str,
    layer: str,
    readout: str,
    reprs: Dict[str, Dict[str, Repr]],
    pairs: pd.DataFrame,
    native_metric: bool = True,
    pca_dims: Optional[int] = None,
) -> Optional[PairFeatures]:
    """Stack one readout over all stimuli, then reduce it to per-pair quantities."""
    stims = stimulus_names(pairs)
    arrays = []
    for stim in stims:
        a = readout_array(reprs[stim][layer], readout)
        if a is None:
            return None
        arrays.append(a)
    X = np.stack(arrays).astype(np.float64)  # (n_stimuli, n_rows, n_cols)
    fps = reprs[stims[0]][layer].fps
    n_rows, n_cols = X.shape[1], X.shape[2]

    if pca_dims is not None:
        # Capacity-matched control: project every model down to the same number of
        # dimensions so that comparisons are not partly a comparison of embedding size.
        # The PCA is fit on the stimuli only -- it never sees a human rating -- so it
        # cannot leak the target, though it does see the held-out stimuli's features.
        flat = X.reshape(len(stims), -1)
        flat = flat - flat.mean(axis=0, keepdims=True)
        k = min(pca_dims, min(flat.shape) - 1)
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
        X = (flat @ vt[:k].T).reshape(len(stims), 1, k)
        n_rows, n_cols = 1, k

    idx = {s: i for i, s in enumerate(stims)}
    a_idx = pairs["ref_stim"].map(idx).to_numpy()
    b_idx = pairs["var_stim"].map(idx).to_numpy()
    diff = X[a_idx] - X[b_idx]  # (n_pairs, n_rows, n_cols)
    sq = diff**2

    if readout == "frames" and pca_dims is None:
        # Average the squared difference over time -> one weight per channel. Keeping
        # every (time, channel) cell separate would let the probe weight individual
        # instants, which fits phase rather than modulation.
        Z = sq.mean(axis=1)
    else:
        # mod_spec deliberately keeps its rows: the point is that different modulation
        # rate bins should be weighted differently.
        Z = sq.reshape(len(pairs), -1)

    zero_shot: Dict[str, np.ndarray] = {}
    if native_metric:
        # Plain L2 in the model's own units -- the distance calc_distances.py computes.
        zero_shot["l2"] = np.sqrt(Z.sum(axis=1))
        fa = X[a_idx].reshape(len(pairs), -1)
        fb = X[b_idx].reshape(len(pairs), -1)
        denom = np.linalg.norm(fa, axis=1) * np.linalg.norm(fb, axis=1) + 1e-12
        zero_shot["cosine"] = 1.0 - (fa * fb).sum(axis=1) / denom
    # L2 after z-scoring each dimension across the stimulus set. Still fit-free, but it
    # removes the "a few high-variance dimensions dominate the norm" failure mode, so it
    # separates "the geometry is wrong" from "the scaling is wrong".
    flat_all = X.reshape(len(stims), -1)
    sd = flat_all.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    dz = (X[a_idx].reshape(len(pairs), -1) - X[b_idx].reshape(len(pairs), -1)) / sd
    zero_shot["l2_z"] = np.linalg.norm(dz, axis=1)

    return PairFeatures(
        model_name, layer, readout, Z, zero_shot, fps, n_rows * n_cols
    )


def build_parameter_baseline(pairs: pd.DataFrame) -> PairFeatures:
    """"You already know the synthesis parameters" baseline.

    If |delta modulation depth| predicts listener distance as well as a 2048-d embedding
    does, the embedding has added nothing beyond identifying which stimulus was played.
    The features are a generic monotone basis of the raw parameter difference, expanded
    per mod type so the probe can learn a different mapping for depth, rate and
    irregularity. It deliberately does NOT use _PERCEPTUAL_WARP -- that would hand the
    baseline the generative model of the simulated ratings.
    """
    d = np.abs(pairs["var_amount"].to_numpy() - pairs["ref_amount"].to_numpy())
    mod_types = pairs["mod_type"].to_numpy()
    d_norm = np.zeros_like(d)
    for mt in np.unique(mod_types):
        mask = mod_types == mt
        span = max(d[mask].max(), 1e-12)
        d_norm[mask] = d[mask] / span

    basis = np.stack([d_norm, np.sqrt(d_norm), d_norm**2, np.log1p(4.0 * d_norm)], axis=1)
    cols = []
    for mt in np.unique(mod_types):
        mask = (mod_types == mt).astype(np.float64)[:, None]
        cols.append(basis * mask)
        cols.append(mask)
    Z = np.concatenate(cols, axis=1)
    return PairFeatures(
        "param_gt", "params", "raw", Z, {"l2": d_norm}, fps=0.0, n_dims=Z.shape[1]
    )


# ======================================================================================
# Part 5 -- Statistics
# ======================================================================================


def _within_block_rank_z(
    values: np.ndarray, block_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Rank within each block, then z-score those ranks within block.

    Correlating the pooled result is a block-controlled rank correlation: it is the
    Spearman correlation with all between-block differences removed. This matters a lot
    here. Pooling raw distances across blocks would let a model score well just by
    knowing that amp pairs are generally further apart than freq pairs, which is a
    statement about the experiment design, not about perceptual distance.

    It is preferred over averaging per-block Spearmans because each block only holds 5
    pairs, so per-block correlations take a handful of discrete values and are extremely
    noisy; pooling the standardised ranks is the same quantity with far less variance.

    Returns (z, valid_mask); entries in constant blocks are invalid.
    """
    z = np.zeros_like(values, dtype=np.float64)
    valid = np.zeros(len(values), dtype=bool)
    for b in np.unique(block_ids):
        mask = block_ids == b
        ranks = stats.rankdata(values[mask])
        sd = ranks.std()
        if sd < 1e-12:
            continue  # Constant block carries no rank information
        z[mask] = (ranks - ranks.mean()) / sd
        valid[mask] = True
    return z, valid


def block_controlled_corr(
    x: np.ndarray, y: np.ndarray, block_ids: np.ndarray
) -> float:
    """Primary agreement statistic: block-controlled Spearman correlation."""
    zx, vx = _within_block_rank_z(x, block_ids)
    zy, vy = _within_block_rank_z(y, block_ids)
    v = vx & vy
    if v.sum() < 3:
        return np.nan
    return float(np.corrcoef(zx[v], zy[v])[0, 1])


def fisher_z_mean_spearman(
    x: np.ndarray, y: np.ndarray, block_ids: np.ndarray
) -> float:
    """Secondary statistic: per-block Spearman averaged in Fisher-z space.

    Reported alongside block_controlled_corr because it is the number most papers
    quote. With 5 pairs per block individual correlations often hit +-1, so they are
    clipped before the atanh to keep the average finite.
    """
    zs = []
    for b in np.unique(block_ids):
        mask = block_ids == b
        if mask.sum() < 3:
            continue
        if x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
            continue  # Constant block: Spearman is undefined, not zero
        r = stats.spearmanr(x[mask], y[mask]).statistic
        if not np.isfinite(r):
            continue
        zs.append(np.arctanh(np.clip(r, -0.9999, 0.9999)))
    if not zs:
        return np.nan
    return float(np.tanh(np.mean(zs)))


def noise_ceiling(
    ratings: np.ndarray, block_ids: np.ndarray, n_splits: int = 500, seed: int = 0
) -> Tuple[float, float, float]:
    """Split-half reliability of the listeners, Spearman-Brown corrected.

    This is the number every model correlation should be read against. Half the
    participants are averaged and correlated against the other half; the correction
    2r/(1+r) extrapolates from half a panel back up to the full panel. A model cannot
    be expected to beat this, because the target it is predicting is itself only this
    reproducible.
    """
    rng = np.random.default_rng(seed)
    n_p = ratings.shape[0]
    vals = []
    for _ in range(n_splits):
        perm = rng.permutation(n_p)
        a = ratings[perm[: n_p // 2]].mean(axis=0)
        b = ratings[perm[n_p // 2 :]].mean(axis=0)
        r = block_controlled_corr(a, b, block_ids)
        if np.isfinite(r):
            sb = 2.0 * r / (1.0 + r) if r > -1.0 else -np.inf
            # A reliability estimate outside [0, 1] is not meaningful; 0 means "this
            # panel carries no reproducible signal at all".
            vals.append(float(np.clip(sb, 0.0, 1.0)))
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    med = float(np.median(vals))
    if med < MIN_USABLE_CEILING:
        log.warning(
            f"Listener noise ceiling is only {med:.3f}. The panel barely agrees with "
            f"itself, so rho_over_ceiling will be unstable and is reported as NaN."
        )
    return med, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


class PrefactorizedRidgeCV:
    """Linear-kernel ridge with nested leave-one-group-out CV, pre-factorised in y.

    Two design decisions worth explaining.

    KERNEL (dual) FORM. The features are squared differences, so p can be tens of
    thousands while n is only 90. Ridge in the dual only ever solves an n x n system, so
    the cost is independent of p. Predictions are identical to the primal solution.

    PRE-FACTORISATION. The fold structure, the standardisation and the kernel all depend
    only on Z, never on the target. So the expensive linear algebra is done once in the
    constructor and each subsequent fit is a handful of matrix-vector products. That is
    what makes a 1000-fold permutation null and a 1000-fold bootstrap affordable -- and
    the permutation null is only valid if the *entire* pipeline, alpha selection
    included, is re-run on each shuffled target, which would otherwise be too slow.

    Grouping is by wavetable: the held-out fold is a timbre the metric has never seen.
    A random split would leak badly, because pairs that share a stimulus are not
    independent and the same modulation levels recur in every block.
    """

    def __init__(self, Z: np.ndarray, groups: np.ndarray, alphas: np.ndarray):
        self.n = Z.shape[0]
        self.alphas = np.asarray(alphas)
        self.folds = []
        p = max(Z.shape[1], 1)

        for g in np.unique(groups):
            te = np.flatnonzero(groups == g)
            tr_ = np.flatnonzero(groups != g)
            # Standardise with training statistics only, so the held-out wavetable never
            # influences the scaling of the features.
            mu = Z[tr_].mean(axis=0)
            sd = Z[tr_].std(axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            Ztr = (Z[tr_] - mu) / sd
            Zte = (Z[te] - mu) / sd
            # Divide by p so the alpha grid means the same thing for a 5-d and a
            # 20000-d feature space.
            Ktr = (Ztr @ Ztr.T) / p
            Kte = (Zte @ Ztr.T) / p

            inner = []
            gtr = groups[tr_]
            for gi in np.unique(gtr):
                ite = np.flatnonzero(gtr == gi)
                itr = np.flatnonzero(gtr != gi)
                Kii = Ktr[np.ix_(itr, itr)]
                Kti = Ktr[np.ix_(ite, itr)]
                eye = np.eye(len(itr))
                ops = [
                    np.linalg.solve(Kii + a * eye, Kti.T).T for a in self.alphas
                ]
                inner.append({"itr": itr, "ite": ite, "ops": ops})

            eye = np.eye(len(tr_))
            outer_ops = [np.linalg.solve(Ktr + a * eye, Kte.T).T for a in self.alphas]
            self.folds.append(
                {"te": te, "tr": tr_, "inner": inner, "outer_ops": outer_ops}
            )

    def cv_predict(self, y: np.ndarray) -> np.ndarray:
        """Out-of-fold predictions, with alpha chosen inside each training fold.

        The alpha chosen per fold is kept in ``self.last_alphas`` so callers can check
        that the grid actually brackets the optimum. An alpha pinned at either end of
        the grid means the grid is mis-specified for these features, and the reported
        correlation is then a property of RIDGE_ALPHAS rather than of the embedding.
        """
        pred = np.full(self.n, np.nan)
        self.last_alphas = np.empty(len(self.folds))
        for fi, fold in enumerate(self.folds):
            ytr = y[fold["tr"]]
            # Inner CV: pick the alpha with the lowest held-out squared error, using
            # only training-fold data.
            errs = np.zeros(len(self.alphas))
            for inner in fold["inner"]:
                y_itr = ytr[inner["itr"]]
                y_ite = ytr[inner["ite"]]
                m = y_itr.mean()
                for ai, op in enumerate(inner["ops"]):
                    p_ite = op @ (y_itr - m) + m
                    errs[ai] += np.sum((p_ite - y_ite) ** 2)
            best = int(np.argmin(errs))
            self.last_alphas[fi] = self.alphas[best]
            m = ytr.mean()
            pred[fold["te"]] = fold["outer_ops"][best] @ (ytr - m) + m
        assert np.isfinite(pred).all()
        return pred

    def warn_if_alpha_at_boundary(self, label: str) -> None:
        """Flag a ridge grid that does not bracket the cross-validated optimum.

        The two ends mean different things and only one of them is a grid problem:

        upper -- the CV wanted maximal shrinkage, i.e. predicting the training mean beat
                 anything the features could do. No finite alpha satisfies that, so
                 widening the grid will not help. This is a RESULT: these features carry
                 no signal that generalises across this particular split. It shows up
                 routinely on leave-one-mod-type-out, where the metric learned on depth
                 and rate has nothing to say about irregularity.
        lower -- the CV wanted no regularisation at all. For a small, well-conditioned
                 feature set that is just ordinary least squares and is fine; for a wide
                 one it means the grid is genuinely mis-specified and should be extended
                 downwards.
        """
        n_folds = len(self.last_alphas)
        hi = int(np.sum(self.last_alphas >= self.alphas.max() * 0.999))
        lo = int(np.sum(self.last_alphas <= self.alphas.min() * 1.001))
        if hi:
            log.info(
                f"{label}: ridge chose maximal shrinkage in {hi}/{n_folds} folds, i.e. "
                f"the training mean beat every fitted metric. Not a grid problem -- "
                f"these features do not generalise across this split."
            )
        if lo:
            log.warning(
                f"{label}: ridge alpha hit the LOWER end of RIDGE_ALPHAS in "
                f"{lo}/{n_folds} folds ({self.alphas.min():.1e}). With "
                f"{self.n} pairs this is unregularised least squares; extend the grid "
                f"downwards if the feature set is wide."
            )


# ======================================================================================
# Part 6 -- Analyses
# ======================================================================================


class HumanData:
    """Aggregated listening test data plus the raw matrix needed for resampling."""

    def __init__(self, ratings: np.ndarray, pairs: pd.DataFrame):
        self.ratings = ratings  # (n_participants, n_pairs)
        self.d = ratings.mean(axis=0)
        self.sem = ratings.std(axis=0, ddof=1) / np.sqrt(ratings.shape[0])
        self.blocks = pairs["block"].to_numpy()
        self.groups = pairs["wavetable"].to_numpy()
        self.n_participants = ratings.shape[0]


def permutation_p(
    observed: float, null: np.ndarray, alternative: str = "greater"
) -> float:
    """One-sided permutation p-value with the +1 correction (never reports p == 0)."""
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(observed):
        return np.nan
    if alternative == "greater":
        n_extreme = int(np.sum(null >= observed))
    else:
        n_extreme = int(np.sum(null <= observed))
    return (1.0 + n_extreme) / (1.0 + len(null))


def permute_within_blocks(
    values: np.ndarray, block_ids: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle values within each block -- the stimulus-label permutation null.

    Because every pair in a block is reference-vs-one-variant, permuting the block's
    values is exactly relabelling which variant produced which human distance. It
    destroys the pairing between stimulus and rating while preserving the block
    structure, the marginal distribution of ratings and the model's own distances. It is
    the only null that accounts for the flexibility of the fitted probe, which is why
    the whole pipeline is re-run on each permuted target.
    """
    out = values.copy()
    for b in np.unique(block_ids):
        mask = block_ids == b
        out[mask] = rng.permutation(values[mask])
    return out


def bootstrap_ci(
    values: np.ndarray, lo: float = 2.5, hi: float = 97.5
) -> Tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def bootstrap_over_wavetables(
    stat_fn, human: HumanData, pairs: pd.DataFrame, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Resample wavetables with replacement.

    The unit of resampling is the wavetable, not the pair: pairs within a block share a
    stimulus and are not exchangeable, so a naive pair bootstrap would give absurdly
    tight intervals. Blocks are relabelled per draw so that a wavetable sampled twice
    contributes two independent blocks rather than one doubled block.
    """
    wts = pairs["wavetable"].to_numpy()
    uniq = np.unique(wts)
    out = np.full(n_boot, np.nan)
    for i in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx_parts, block_parts = [], []
        for rep, wt in enumerate(drawn):
            sel = np.flatnonzero(wts == wt)
            idx_parts.append(sel)
            block_parts.append([f"{b}__rep{rep}" for b in human.blocks[sel]])
        idx = np.concatenate(idx_parts)
        blocks = np.concatenate(block_parts)
        out[i] = stat_fn(idx, blocks, human.d)
    return out


def bootstrap_over_participants(
    stat_fn, human: HumanData, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Resample participants with replacement -- how much of the estimate is panel luck."""
    all_idx = np.arange(len(human.d))
    out = np.full(n_boot, np.nan)
    for i in range(n_boot):
        who = rng.integers(0, human.n_participants, human.n_participants)
        out[i] = stat_fn(all_idx, human.blocks, human.ratings[who].mean(axis=0))
    return out


def _score_signs(pred_sign: np.ndarray, true_sign: np.ndarray) -> np.ndarray:
    """1 for a correct ordering, 0 for a wrong one, 0.5 for a tie (i.e. a guess)."""
    return np.where(pred_sign == 0, 0.5, (pred_sign == true_sign).astype(float))


def ordinal_agreement(
    model_d: np.ndarray, human: HumanData, alpha: float = 0.05, n_splits: int = 200,
    seed: int = 0,
) -> Tuple[float, int, float]:
    """Fraction of reliable listener orderings the model reproduces.

    Within a block every pair is reference-vs-variant, so comparing two pairs asks
    "is variant x or variant y further from the reference?" -- a triplet judgement.
    Only comparisons where the panel is reliable are counted: a paired t-test across
    participants (the same people rated both pairs) must reject at alpha. Without this
    gate the score is dominated by comparisons where listeners themselves are guessing.

    Returns (model accuracy, number of gated comparisons, listener ceiling). The ceiling
    is how often a random half of the panel reproduces the other half's ordering on the
    same comparisons, so it plays the same role as the split-half noise ceiling.
    """
    rng = np.random.default_rng(seed)
    gated: List[Tuple[int, int]] = []
    for b in np.unique(human.blocks):
        idx = np.flatnonzero(human.blocks == b)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                p, q = idx[i], idx[j]
                diff = human.ratings[:, p] - human.ratings[:, q]
                if diff.std(ddof=1) < 1e-12:
                    continue
                t_res = stats.ttest_1samp(diff, 0.0)
                if t_res.pvalue < alpha:
                    gated.append((p, q))
    if not gated:
        return np.nan, 0, np.nan

    p_idx = np.array([g[0] for g in gated])
    q_idx = np.array([g[1] for g in gated])
    human_sign = np.sign(human.d[p_idx] - human.d[q_idx])
    model_sign = np.sign(model_d[p_idx] - model_d[q_idx])
    acc = float(np.mean(_score_signs(model_sign, human_sign)))

    n_p = human.n_participants
    ceil = []
    for _ in range(n_splits):
        perm = rng.permutation(n_p)
        a = human.ratings[perm[: n_p // 2]].mean(axis=0)
        b = human.ratings[perm[n_p // 2 :]].mean(axis=0)
        ceil.append(
            np.mean(
                _score_signs(np.sign(a[p_idx] - a[q_idx]), np.sign(b[p_idx] - b[q_idx]))
            )
        )
    return acc, len(gated), float(np.mean(ceil))


def analyse(
    pf: PairFeatures,
    human: HumanData,
    pairs: pd.DataFrame,
    ceiling: float,
    rng: np.random.Generator,
    store_predictions: Optional[Dict[str, np.ndarray]] = None,
) -> List[dict]:
    """Run every analysis for one (model, layer, readout) and return tidy result rows."""
    rows = []
    # Normalising by a near-zero ceiling turns a small correlation into a huge ratio.
    usable_ceiling = ceiling if ceiling >= MIN_USABLE_CEILING else np.nan

    base = {
        "model": pf.model,
        "layer": pf.layer,
        "readout": pf.readout,
        "n_dims": pf.n_dims,
        "fps": pf.fps,
        "covers_max_mod_rate": pf.covers_rate,
        "n_pairs": len(pairs),
        "ceiling": ceiling,
    }

    # --- Analysis 1: zero-shot, no fitting whatsoever -------------------------------
    for metric_name, d_model in pf.zero_shot.items():
        # Human distance grows with dissimilarity and so does every metric here, so a
        # positive correlation is the hypothesis and "greater" is the right tail.
        def stat_fn(idx, blocks, human_d, _d=d_model):
            return block_controlled_corr(_d[idx], human_d[idx], blocks)

        obs = block_controlled_corr(d_model, human.d, human.blocks)
        null = np.array(
            [
                block_controlled_corr(
                    d_model, permute_within_blocks(human.d, human.blocks, rng), human.blocks
                )
                for _ in range(N_PERMUTATIONS)
            ]
        )
        boot_wt = bootstrap_over_wavetables(stat_fn, human, pairs, N_BOOTSTRAPS, rng)
        boot_pp = bootstrap_over_participants(stat_fn, human, N_BOOTSTRAPS, rng)
        lo_wt, hi_wt = bootstrap_ci(boot_wt)
        lo_pp, hi_pp = bootstrap_ci(boot_pp)
        acc, n_gated, acc_ceiling = ordinal_agreement(d_model, human)
        rows.append(
            {
                **base,
                "analysis": "zero_shot",
                "metric": metric_name,
                "rho": obs,
                "rho_over_ceiling": obs / usable_ceiling,
                "rho_fisher_z_mean": fisher_z_mean_spearman(d_model, human.d, human.blocks),
                "p_perm": permutation_p(obs, null),
                "ci_lo_wavetable": lo_wt,
                "ci_hi_wavetable": hi_wt,
                "ci_lo_participant": lo_pp,
                "ci_hi_participant": hi_pp,
                "ordinal_acc": acc,
                "ordinal_n": n_gated,
                "ordinal_ceiling": acc_ceiling,
            }
        )

    # --- Analysis 2: the probe ------------------------------------------------------
    # A diagonal Mahalanobis metric fitted on the squared differences, cross-validated
    # across wavetables. This is the "is the information in there at all" test.
    probe = PrefactorizedRidgeCV(pf.Z, human.groups, RIDGE_ALPHAS)
    pred = probe.cv_predict(human.d)
    probe.warn_if_alpha_at_boundary(f"{pf.key} (leave-one-wavetable-out)")
    obs = block_controlled_corr(pred, human.d, human.blocks)

    # The null re-fits the probe (alpha selection included) on each shuffled target, so
    # any correlation the fitting procedure can manufacture from noise shows up here.
    null = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        y_perm = permute_within_blocks(human.d, human.blocks, rng)
        null[i] = block_controlled_corr(probe.cv_predict(y_perm), y_perm, human.blocks)

    def probe_stat(idx, blocks, human_d):
        # Refit on the resampled target, then score on the requested subset. Cheap
        # because the ridge operators are pre-factorised. The wavetable bootstrap passes
        # the original target unchanged, in which case the fit is reused as-is.
        p = pred if human_d is human.d else probe.cv_predict(human_d)
        return block_controlled_corr(p[idx], human_d[idx], blocks)

    boot_wt = bootstrap_over_wavetables(probe_stat, human, pairs, N_BOOTSTRAPS, rng)
    boot_pp = bootstrap_over_participants(probe_stat, human, N_BOOTSTRAPS, rng)
    lo_wt, hi_wt = bootstrap_ci(boot_wt)
    lo_pp, hi_pp = bootstrap_ci(boot_pp)
    acc, n_gated, acc_ceiling = ordinal_agreement(pred, human)

    if store_predictions is not None:
        store_predictions[pf.key] = pred

    rows.append(
        {
            **base,
            "analysis": "probe",
            "metric": "diag_mahalanobis_logo_wavetable",
            "rho": obs,
            "rho_over_ceiling": obs / usable_ceiling,
            "rho_fisher_z_mean": fisher_z_mean_spearman(pred, human.d, human.blocks),
            "p_perm": permutation_p(obs, null),
            "ci_lo_wavetable": lo_wt,
            "ci_hi_wavetable": hi_wt,
            "ci_lo_participant": lo_pp,
            "ci_hi_participant": hi_pp,
            "ordinal_acc": acc,
            "ordinal_n": n_gated,
            "ordinal_ceiling": acc_ceiling,
        }
    )

    # --- Analysis 3: the harder generalisation split --------------------------------
    # Holding out a whole modulation type asks whether the learned weighting is about
    # perceptual distance in general or just about depth/rate/irregularity separately.
    # A NaN here is meaningful, not a failure: it means the fitted metric predicts a
    # constant for the held-out modulation type, so it has no ranking information at
    # all. The parameter baseline does exactly this, because its features are one-hot
    # expanded per mod type and every column is zero outside the training folds.
    probe_mt = PrefactorizedRidgeCV(pf.Z, pairs["mod_type"].to_numpy(), RIDGE_ALPHAS)
    pred_mt = probe_mt.cv_predict(human.d)
    probe_mt.warn_if_alpha_at_boundary(f"{pf.key} (leave-one-mod-type-out)")
    obs_mt = block_controlled_corr(pred_mt, human.d, human.blocks)
    null_mt = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        y_perm = permute_within_blocks(human.d, human.blocks, rng)
        null_mt[i] = block_controlled_corr(
            probe_mt.cv_predict(y_perm), y_perm, human.blocks
        )
    acc, n_gated, acc_ceiling = ordinal_agreement(pred_mt, human)
    rows.append(
        {
            **base,
            "analysis": "probe_logo_mod_type",
            "metric": "diag_mahalanobis_logo_mod_type",
            "rho": obs_mt,
            "rho_over_ceiling": obs_mt / usable_ceiling,
            "rho_fisher_z_mean": fisher_z_mean_spearman(pred_mt, human.d, human.blocks),
            "p_perm": permutation_p(obs_mt, null_mt),
            "ci_lo_wavetable": np.nan,
            "ci_hi_wavetable": np.nan,
            "ci_lo_participant": np.nan,
            "ci_hi_participant": np.nan,
            "ordinal_acc": acc,
            "ordinal_n": n_gated,
            "ordinal_ceiling": acc_ceiling,
        }
    )
    return rows


# ======================================================================================
# Part 7 -- Plots
# ======================================================================================
# Styling follows calc_distances.py so the figures sit next to the distance curves.

READOUT_COLORS = {"time_avg": "#b0562a", "frames": SERIES_COLOR, "mod_spec": "#2aa06a"}


def _style(ax) -> None:
    ax.grid(True, color=AXIS_COLOR, alpha=0.15, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _layer_depth(layer: str) -> int:
    """Order layers along the network. The clip embedding sits after every conv block."""
    if layer.startswith("conv_block"):
        return int(layer.replace("conv_block", ""))
    return 99


def plot_summary(df: pd.DataFrame, ceiling: float, save_dir: str) -> None:
    """Best zero-shot vs probe per model, against the listener noise ceiling.

    Two bars per model on purpose: the gap between them is the whole point. A short
    zero-shot bar with a tall probe bar means the information is present but the default
    L2 does not expose it -- a fixable problem, e.g. by learning a metric on top or by
    changing the pooling. Both short means it is not there.

    The two bars are read off the SAME (layer, readout), namely the one where the probe
    does best, so the gap is a like-for-like comparison rather than the best of one
    thing against the best of a different thing. Note that picking the best layer is a
    selection over many comparisons; the per-row p_perm values in results.csv are not
    corrected for it.
    """
    best = []
    for model, g in df.groupby("model", sort=False):
        pr = g[(g["analysis"] == "probe") & g["rho"].notna()]
        if not len(pr):
            continue
        top = pr.loc[pr["rho"].idxmax()]
        zs = g[
            (g["analysis"] == "zero_shot")
            & (g["layer"] == top["layer"])
            & (g["readout"] == top["readout"])
        ]["rho"]
        best.append(
            {
                "model": model,
                "label": f"{model}\n{top['layer']} / {top['readout']}",
                "zero_shot": zs.max() if len(zs) else np.nan,
                "probe": top["rho"],
            }
        )
    if not best:
        return
    best = pd.DataFrame(best).sort_values("probe", ascending=True)

    y = np.arange(len(best))
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(best) + 2.5))
    ax.barh(y - 0.2, best["zero_shot"], height=0.38, color="#b0562a",
            label="zero-shot (best of L2 / cosine / z-scored L2)")
    ax.barh(y + 0.2, best["probe"], height=0.38, color=SERIES_COLOR,
            label="probe (diagonal Mahalanobis, leave-one-wavetable-out)")
    ax.axvline(ceiling, color=AXIS_COLOR, linestyle="--", linewidth=1.2,
               label=f"listener noise ceiling = {ceiling:.2f}")
    ax.axvline(0.0, color=AXIS_COLOR, linewidth=0.8, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(best["label"], fontsize=7)
    ax.set_xlabel("block-controlled Spearman with human distance")
    ax.set_title("How much human perceptual distance does each embedding carry?\n"
                 "(each model shown at its best-probing layer and readout)", fontsize=11)
    # Below the axes so it cannot sit on top of the shortest bar.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False,
              fontsize=7, labelcolor=AXIS_COLOR)
    _style(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary.png"), dpi=DPI)
    plt.close(fig)


def plot_layerwise(df: pd.DataFrame, ceiling: float, save_dir: str) -> None:
    """Probe correlation as a function of depth, one panel per network.

    The expected shape is a hump: perceptual information peaks somewhere mid-network and
    is discarded by the layers specialised for AudioSet classification. Layers whose
    frame rate cannot represent a 4 Hz modulation are marked, because a low score there
    is explained by temporal downsampling rather than by the representation.
    """
    probes = df[(df["analysis"] == "probe") & df["layer"].str.startswith("conv_block")]
    # The capacity-matched controls only ever cover a single layer, so they would show
    # up as an empty one-point panel here.
    depth_counts = probes.groupby("model")["layer"].nunique()
    models = sorted(depth_counts[depth_counts > 1].index)
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), squeeze=False)
    for ax, model in zip(axes[0], models):
        g = probes[probes["model"] == model]
        for readout, gg in g.groupby("readout"):
            gg = gg.assign(depth=gg["layer"].map(_layer_depth)).sort_values("depth")
            ax.plot(gg["depth"], gg["rho"], marker="o", linewidth=2.0,
                    color=READOUT_COLORS.get(readout, AXIS_COLOR), label=readout)
            # Hollow markers where the layer cannot represent the fastest modulation.
            bad = gg[gg["covers_max_mod_rate"] == False]  # noqa: E712 - NaN means n/a
            ax.scatter(bad["depth"], bad["rho"], s=90, facecolors="none",
                       edgecolors=AXIS_COLOR, linewidths=1.5, zorder=3)
        ax.axhline(ceiling, color=AXIS_COLOR, linestyle="--", linewidth=1.2)
        ax.axhline(0.0, color=AXIS_COLOR, linewidth=0.8, alpha=0.5)
        ax.set_xlabel("conv block")
        ax.set_ylabel("probe rho with human distance")
        ax.set_title(f"{model}\n(hollow = Nyquist below {MAX_STIMULUS_MOD_RATE_HZ} Hz)", fontsize=9)
        ax.legend(loc="best", frameon=False, fontsize=8, labelcolor=AXIS_COLOR)
        _style(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "layerwise.png"), dpi=DPI)
    plt.close(fig)


def plot_best_scatter(
    pred: np.ndarray, human: HumanData, pairs: pd.DataFrame, title: str, save_dir: str
) -> None:
    """Predicted vs actual human distance for the best probe, split by modulation type.

    Two panels, because the probe is fitted by least squares but scored by rank:

      left  -- raw predictions. The scale is NOT calibrated to the rating scale (ridge
               extrapolating onto an unseen wavetable routinely overshoots), so there is
               deliberately no identity line here. What to look at is the spread and
               whether any modulation type is systematically off.
      right -- within-block ranks, which is exactly what block_controlled_corr consumes.
               This is the panel that corresponds to the reported number, and points
               should sit on the diagonal.

    Worth looking at even when the correlation is high: it shows whether agreement comes
    from all three manipulations or from one of them carrying the whole result.
    """
    colors = {"amp": SERIES_COLOR, "freq": "#b0562a", "reg": "#2aa06a"}
    rank_pred, _ = _within_block_rank_z(pred, human.blocks)
    rank_human, _ = _within_block_rank_z(human.d, human.blocks)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for mt, g in pairs.groupby("mod_type"):
        idx = g.index.to_numpy()
        axes[0].errorbar(human.d[idx], pred[idx], xerr=human.sem[idx], fmt="o",
                         markersize=6, alpha=0.8, color=colors.get(mt, AXIS_COLOR),
                         elinewidth=1.0, capsize=2, label=mt)
        axes[1].scatter(rank_human[idx], rank_pred[idx], s=36, alpha=0.8,
                        color=colors.get(mt, AXIS_COLOR), label=mt)

    axes[0].set_xlabel("human distance (mean rating, x-error = SEM)")
    axes[0].set_ylabel("probe prediction (uncalibrated units)")
    axes[0].set_title("raw predictions", fontsize=9)

    lims = [
        min(np.nanmin(rank_human), np.nanmin(rank_pred)),
        max(np.nanmax(rank_human), np.nanmax(rank_pred)),
    ]
    axes[1].plot(lims, lims, color=AXIS_COLOR, linestyle="--", linewidth=1.0, alpha=0.5)
    axes[1].set_xlabel("human distance, within-block rank (z)")
    axes[1].set_ylabel("probe prediction, within-block rank (z)")
    axes[1].set_title("within-block ranks (what the reported rho measures)", fontsize=9)

    for ax in axes:
        ax.set_box_aspect(1)
        ax.legend(loc="best", frameon=False, fontsize=8, labelcolor=AXIS_COLOR)
        _style(ax)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "best_probe_scatter.png"), dpi=DPI)
    plt.close(fig)


# ======================================================================================
# Part 8 -- Entry point
# ======================================================================================


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    rng = np.random.default_rng(ANALYSIS_SEED)

    # --- Stimuli, pairs and listeners ------------------------------------------------
    pairs = build_pairs()
    pairs.to_csv(os.path.join(SAVE_DIR, "pairs.csv"), index=False)

    if HUMAN_RATINGS_CSV:
        long = load_human_ratings(HUMAN_RATINGS_CSV, pairs)
    else:
        long = simulate_human_ratings(pairs)
        placeholder_path = os.path.join(SAVE_DIR, "human_ratings_placeholder.csv")
        long.to_csv(placeholder_path, index=False)
        log.info(f"Wrote placeholder ratings to {placeholder_path}")

    ratings = ratings_to_matrix(long, pairs)
    human = HumanData(ratings, pairs)
    ceil_med, ceil_lo, ceil_hi = noise_ceiling(ratings, human.blocks)
    log.info(
        f"Listener noise ceiling (split-half, Spearman-Brown corrected): "
        f"{ceil_med:.3f} [{ceil_lo:.3f}, {ceil_hi:.3f}]. No model should be expected "
        f"to exceed this."
    )

    results: List[dict] = []
    predictions: Dict[str, np.ndarray] = {}

    # --- Control that needs no model: the synthesis parameters themselves -------------
    log.info("Analysing the ground-truth parameter baseline")
    results += analyse(build_parameter_baseline(pairs), human, pairs, ceil_med, rng, predictions)

    # --- Models ----------------------------------------------------------------------
    # Each entry is (label, factory). Factories are lazy so that one model failing to
    # download or import does not take the rest of the analysis with it.
    factories = [
        ("lowlevel", lambda: LowLevelFeatureExtractor()),
        ("panns", lambda: PANNsExtractor("cnn14-32k")),
        # The random-weights control. If this matches the trained model, the result is
        # about architecture and the mel front end, not about what training learned.
        ("panns_random", lambda: PANNsExtractor("cnn14-32k", randomise=True, seed=0)),
        ("clap", lambda: ClapExtractor(use_cuda=False)),
    ]

    stims = stimulus_names(pairs)
    for label, factory in factories:
        try:
            extractor = factory()
            reprs = extract_all(extractor, stims)
        except Exception as e:  # noqa: BLE001 - a missing checkpoint should not be fatal
            log.warning(f"Skipping {label}: {type(e).__name__}: {e}")
            continue

        layers = list(reprs[stims[0]].keys())
        model_rows: List[dict] = []
        for layer in layers:
            r0 = reprs[stims[0]][layer]
            for readout in READOUTS:
                if r0.is_clip_level and readout == "frames":
                    continue  # Identical to time_avg when there is only one frame
                pf = build_pair_features(
                    extractor.name, layer, readout, reprs, pairs, extractor.native_metric
                )
                if pf is None:
                    log.info(
                        f"{extractor.name}.{layer}: readout '{readout}' unavailable "
                        f"(no usable time axis)"
                    )
                    continue
                log.info(f"Analysing {pf.key} ({pf.n_dims} features)")
                model_rows += analyse(pf, human, pairs, ceil_med, rng, predictions)
        results += model_rows

        # --- Capacity-matched control ---------------------------------------------
        # Re-run this model's strongest representation after projecting it to a fixed
        # number of PCA dimensions, so that comparisons across models are not partly a
        # comparison of embedding size. Picking the strongest representation is a
        # selection on the data, so treat this as a control, not as a headline number.
        probe_rows = [r for r in model_rows if r["analysis"] == "probe" and np.isfinite(r["rho"])]
        if probe_rows:
            top = max(probe_rows, key=lambda r: r["rho"])
            pf = build_pair_features(
                f"{extractor.name}__pca{PCA_MATCH_DIMS}", top["layer"], top["readout"],
                reprs, pairs, extractor.native_metric, pca_dims=PCA_MATCH_DIMS,
            )
            if pf is not None:
                log.info(f"Analysing capacity-matched control {pf.key}")
                results += analyse(pf, human, pairs, ceil_med, rng, predictions)

    # --- Save ------------------------------------------------------------------------
    df = pd.DataFrame(results)
    df["ceiling_lo"] = ceil_lo
    df["ceiling_hi"] = ceil_hi
    df["human_data"] = "PLACEHOLDER" if not HUMAN_RATINGS_CSV else HUMAN_RATINGS_CSV
    csv_path = os.path.join(SAVE_DIR, "results.csv")
    df.to_csv(csv_path, index=False)
    log.info(f"Saved {len(df)} result rows to {csv_path}")

    plot_summary(df, ceil_med, SAVE_DIR)
    plot_layerwise(df, ceil_med, SAVE_DIR)
    probes = df[df["analysis"] == "probe"]
    if len(probes) and probes["rho"].notna().any():
        top = probes.loc[probes["rho"].idxmax()]
        key = f"{top['model']}|{top['layer']}|{top['readout']}"
        if key in predictions:
            plot_best_scatter(
                predictions[key], human, pairs,
                f"{key}\nprobe rho = {top['rho']:.3f} (ceiling {ceil_med:.3f})",
                SAVE_DIR,
            )

    # --- Console summary --------------------------------------------------------------
    log.info("\n" + _format_summary(df, ceil_med))


def _covers_str(value) -> str:
    """Nyquist coverage as text; NaN means the question does not apply to this readout."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return str(bool(value))


def _format_summary(df: pd.DataFrame, ceiling: float) -> str:
    """A readable leaderboard, sorted by the probe result."""
    lines = [
        f"{'model|layer|readout':<52} {'zero-shot':>10} {'probe':>8} {'/ceil':>7} "
        f"{'p_perm':>8} {'ordinal':>8} {'covers4Hz':>10}"
    ]
    probes = df[df["analysis"] == "probe"].sort_values("rho", ascending=False)
    for _, r in probes.iterrows():
        zs = df[
            (df["model"] == r["model"])
            & (df["layer"] == r["layer"])
            & (df["readout"] == r["readout"])
            & (df["analysis"] == "zero_shot")
        ]["rho"]
        zs_best = zs.max() if len(zs) else np.nan
        key = f"{r['model']}|{r['layer']}|{r['readout']}"
        lines.append(
            f"{key:<52} {zs_best:>10.3f} {r['rho']:>8.3f} "
            f"{r['rho'] / ceiling if ceiling >= MIN_USABLE_CEILING else np.nan:>7.2f} "
            f"{r['p_perm']:>8.4f} "
            f"{r['ordinal_acc']:>8.3f} {_covers_str(r['covers_max_mod_rate']):>10}"
        )
    lines.append(f"\nlistener noise ceiling: {ceiling:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
