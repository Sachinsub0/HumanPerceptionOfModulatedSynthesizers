"""Do the neural audio embeddings encode human perceptual distance?

Takes the exact reference-vs-variant pairs that ``calc_distances.py`` measures and asks
how much of the *listener* dissimilarity each representation accounts for. Two claims,
kept separate because only the first one is answered by a distance function alone:

  ZERO-SHOT  The off-the-shelf distance (L2 on the embedding) ranks pairs the way
             listeners do. This is the claim that matters if you want to *use* the
             embedding as a loss. Nothing is fitted, so it is fully falsifiable.
  PROBE      Perceptual dissimilarity is linearly decodable from the embedding even
             when plain L2 fails. Strictly secondary: at this N a probe costs about an
             order of magnitude in statistical power (see the note on N below).

WHY THE DESIGN FORCES THESE CHOICES
-----------------------------------
The listening test gives 6 wavetables x 3 mod types x 5 amounts = 90 rated stimuli,
each rated by 50 participants (4500 observations). Three facts drive everything here:

1. N IS 72, NOT 4500. The 50 ratings of a stimulus are replicates of one identical
   feature vector; they reduce noise in y but add no information about the x -> y map.
   A leave-one-out CV over the 4500 rows leaks catastrophically: hold out one rating and
   the other 49 ratings of the *same* stimulus stay in training, so the model memorises
   the 90 stimulus means instead of generalising. On simulated data with a pure-noise
   embedding that scheme reports rho = +0.86 against individual ratings and +0.98
   against stimulus means. Every CV here therefore holds out whole WAVETABLES or whole
   MOD TYPES, never rows and never single stimuli.
2. THE 18 SELF-PAIRS ARE A FREE WIN. Each block's least-modulated stimulus is both the
   reference and one of the 5 rated stimuli, so 18 of the 90 pairs are ref-vs-itself
   where every distance is exactly 0. Those 18 points alone buy rho = +0.73 for a
   noise embedding. They are kept for participant screening (they are hidden-reference
   trials) and excluded from the primary statistics; see DROP_SELF_PAIRS.
3. CHANCE IS NOT ZERO. All 4 pairs in a block share one reference embedding, so pair
   features leak block identity, and the block structure of the stimuli means a
   statistic can be non-zero under the null. Measured, not assumed: every statistic
   gets a permutation null obtained by re-running the whole pipeline (PCA and alpha
   selection included) on shuffled targets, and the null mean is reported next to the
   observed value.

Everything is also reported relative to a NOISE CEILING estimated per statistic by
split-half reliability of the listeners themselves. rho = 0.55 against a ceiling of 0.60
is essentially perfect; the same number against a ceiling of 0.95 is weak.

THE LISTENING TEST DATA HERE IS NOISE
-------------------------------------
The real ratings are not in the repo, so HUMAN_MODE controls what stands in:

  "noise"      (default) ratings are drawn independently of the stimulus. There is no
               signal to find, so every statistic must land at its permutation null and
               the noise ceiling must collapse to ~0. This is the pipeline's NULL TEST:
               if anything here reports a significant result on noise, the analysis is
               broken. Expect rho/ceiling to be reported as NaN, and the participant
               screen to flag nearly everyone -- both are correct behaviour.
  "simulated"  ratings are monotone-saturating in modulation amount with block-specific
               sensitivity. A POSITIVE CONTROL: use it to check the pipeline detects
               signal when signal exists. Do not read the numbers as a result -- they
               are generated from the synthesis parameters, so the parameter baseline is
               at ceiling by construction.
  "csv"        real data from HUMAN_RATINGS_CSV, tidy columns:
                   participant, wavetable, mod_type, amount, rating
               Nothing else in the script changes.

USAGE
-----
    python scripts/neural_analysis_2.py                    # full run, noise placeholder
    QUICK=1 python scripts/neural_analysis_2.py            # ~20x fewer permutations
    HUMAN_MODE=simulated python scripts/neural_analysis_2.py  # positive control

The two placeholder modes are the pipeline's own test suite. On "noise" nothing may clear
its permutation null and the ceiling must collapse; on "simulated" the oracle, the real
distance functions and the synthetic_signal control must all be detected while noise_768
must not. Both are quick enough to re-run after any change to the statistics.

Real model embeddings are read from out/embeddings/<model>/<stimulus>.npz. Set
EXTRACT_MISSING_EMBEDDINGS = True to build that cache from code/losses.py (needs the
CLAP / PANNs checkpoints). Any model without a cache is skipped with a warning, so the
script always runs: the synthetic controls and the distance CSVs from calc_distances.py
are enough to exercise every statistic.

Outputs land in out/neural_analysis/: pairs.csv, human_ratings.csv, zero_shot.csv,
probe.csv, sensitivity.csv and three figures.
"""

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # Headless: the script only ever writes PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
# code/ is a source root in the IDE but not on sys.path for a plain `python scripts/...`
for _p in (SCRIPT_DIR, os.path.join(REPO_DIR, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the pair enumeration helpers that calc_distances.py uses rather than
# re-deriving them, so the two scripts cannot drift apart on which pairs are compared.
from util import find_variants, parse_amount

# scipy warns when a correlation gets constant input. That happens legitimately here
# (e.g. a degenerate bootstrap draw), the statistic comes back NaN, and NaNs are dropped
# downstream -- so the warning is noise rather than information.
for _warn in ("ConstantInputWarning", "NearConstantInputWarning"):
    _cls = getattr(stats, _warn, None)
    if _cls is not None:
        warnings.filterwarnings("ignore", category=_cls)

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# Matches the plot styling in calc_distances.py
SERIES_COLOR = "#2a78d6"
CONTRAST_COLOR = "#d6552a"
AXIS_COLOR = "#52514e"
DPI = 150


# ======================================================================================
# Configuration
# ======================================================================================

SAMPLES_DIR = os.path.join(REPO_DIR, "out", "samples_all")
EMB_DIR = os.path.join(REPO_DIR, "out", "embeddings")
DIST_DIR = os.path.join(REPO_DIR, "out", "distances")
SAVE_DIR = os.path.join(REPO_DIR, "out", "neural_analysis")

SR = 44100
TARGET_LUFS = -18
SUFFIX = f"_{TARGET_LUFS}lufs.wav"

# Same wavetables and reference modulations as calc_distances.py
WAVETABLES = [
    "brightness_real__harmonics__synced_sines__256_1024",
    "brightness_synthetic__256_1024",
    "richness_real__filter__acid_saw__46_1024__inverted",
    "richness_synthetic__256_1024",
    "warmth_real__vintage__logue_saw__166_1024",
    "warmth_synthetic__256_1024",
]
# The least-modulated stimulus of each mod type, which is what listeners compared against
MOD_SIG_REFERENCES = [
    "amp_1.00hz_0.10",
    "freq_0.25hz",
    "reg_1.00hz_0.000",
]

# --- Listening test ------------------------------------------------------------------
# Overridable from the shell: HUMAN_MODE=simulated python scripts/neural_analysis_2.py
HUMAN_MODE = os.environ.get("HUMAN_MODE", "noise")  # "noise"|"simulated"|"csv"
HUMAN_RATINGS_CSV: Optional[str] = None  # required when HUMAN_MODE == "csv"
N_PARTICIPANTS = 50
RATING_MIN, RATING_MAX = 0.0, 100.0
HUMAN_SEED = 0

# Participant screening on the 18 hidden-reference trials (ref vs itself, true answer 0).
# "report" flags without dropping, which is the safe default: inspect the flags before
# you let a screen delete data. Switch to "drop" once you have looked at real ratings.
SCREEN_ACTION = "report"  # "report" | "drop"
SCREEN_MAX_HIDDEN_REF = 20.0  # Median hidden-reference rating above this is suspicious
SCREEN_MIN_AMOUNT_RHO = 0.0  # Within-block rank corr with amount at or below -> flagged
SCREEN_MAX_DROP_FRAC = 0.5  # Refuse to drop more than this fraction of participants

# Primary analysis excludes the trivial self pairs (see docstring point 2). The
# sensitivity table re-runs the point estimates with them kept.
DROP_SELF_PAIRS = True
# Primary uses raw per-stimulus means; rank statistics are robust to how individual
# participants use the scale. The sensitivity table re-runs with per-participant z-scores.
PARTICIPANT_SCALING = "raw"  # "raw" | "zscore"

# --- Inference -----------------------------------------------------------------------
QUICK = os.environ.get("QUICK", "0") == "1"
N_SPLIT_HALVES = 200 if QUICK else 1000  # Noise ceiling splits
N_PERMUTATIONS = 100 if QUICK else 2000  # Zero-shot permutation nulls
N_BOOTSTRAPS = 100 if QUICK else 2000  # Cluster bootstrap CIs
N_PERM_PROBE = 25 if QUICK else 200  # Probe nulls: each one re-fits the whole probe
# Below this, dividing by the ceiling amplifies noise more than it corrects for it, so
# the ratio is reported as NaN. With HUMAN_MODE="noise" this triggers by design.
MIN_USABLE_CEILING = 0.2
# Reported alongside every model as the "did you need a network at all" reference point
COMPARISON_BASELINE = "amount_rank"
ANALYSIS_SEED = 42

# --- Probe ---------------------------------------------------------------------------
# Unsupervised PCA down to this many dims before the ridge. Fitted inside the training
# fold, on the embeddings only -- no labels touched. 72 points cannot support a probe on
# 768 raw dims (measured: p = 0.06 even when the signal is genuinely present).
PROBE_PCA_DIMS = 16
# Reported both ways, because PCA is not automatically the right reducer: it keeps the
# highest-variance directions, so if the perceptual signal lives in a few dimensions of
# an otherwise isotropic feature set it gets discarded. Real embeddings are far from
# isotropic (a handful of components carry most of the variance) so PCA usually helps
# there, but "usually" is not an argument -- run both and show both. None = no PCA,
# ridge on all standardised dimensions.
PROBE_DIM_VARIANTS: List[Optional[int]] = [16, None]
# Must bracket the CV optimum at both ends. The upper end matters because holding out a
# whole wavetable often makes "shrink to the training mean" the best available fit.
RIDGE_ALPHAS = np.logspace(-3, 8, 23)
# Outer CV schemes. Never single stimuli and never rows: see docstring point 1.
PROBE_SCHEMES = ["leave_one_wavetable_out", "leave_one_mod_type_out"]

# --- Representations -----------------------------------------------------------------
# Real models, read from out/embeddings/<name>/. "__randinit" is the control that
# separates a learned representation from the architecture and its filterbank.
EMBEDDING_MODELS = [
    "clap_2023",
    "panns_cnn14_32k",
    "panns_wavegram_logmel",
    "clap_2023__randinit",
    "panns_cnn14_32k__randinit",
]
# Distances already computed by calc_distances.py (mse, mss, jtfs, ...). Zero-shot only:
# a scalar distance cannot feed the probe.
DISTANCE_CSVS = ["distances.csv", "distances_jtfs.csv"]
# Set True to build the embedding cache with code/losses.py (needs the checkpoints)
EXTRACT_MISSING_EMBEDDINGS = False
# Canonical zero-shot metric, i.e. the one you would actually use as a loss. The others
# are labelled as sensitivity rows so the multiplicity is explicit.
CANONICAL_METRIC = "l2"
SENSITIVITY_METRICS = ["cosine", "whitened_l2"]
# Readout applied to a cached embedding before distances are taken. "mean" is the
# canonical clip-level pooling that EmbeddingLoss.forward already does. The others need
# a framewise (2-D) cache and target the fact that mean pooling is nearly blind to
# modulation RATE, which is exactly what the freq condition manipulates.
CANONICAL_READOUT = "mean"
SENSITIVITY_READOUTS = ["mean_std", "modspec"]
MOD_SPEC_FMIN_HZ, MOD_SPEC_FMAX_HZ = 0.1, 20.0  # Stimuli modulate at 0.25-4 Hz


# ======================================================================================
# Part 1 -- Stimuli and pairs
# ======================================================================================


def build_pairs() -> pd.DataFrame:
    """Enumerate exactly the pairs that calc_distances.py measures.

    For each wavetable and each reference modulation, find every variant that differs
    only in its amount: 6 wavetables x 3 mod types x 5 levels = 90 pairs, 18 of which
    are the trivial self comparison.

    A "block" is one (wavetable, mod_type) cell and is the unit of analysis throughout.
    Blocks matter because pooling across them lets a trivial "amp pairs are all bigger
    than freq pairs" main effect masquerade as perceptual agreement.
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
                        "mod_sig": var_mod_sig,
                        "ref_amount": ref_amount,
                        "amount": var_amount,
                        "is_self": var_amount == ref_amount,
                    }
                )
    df = pd.DataFrame(rows)
    # Ordinal position of the amount within its block, 1..5. Used as the parameter
    # oracle baseline: any representation that cannot beat "rank of the swept parameter"
    # has told us nothing beyond "more modulation sounds more different".
    df["amount_rank"] = df.groupby("block")["amount"].rank(method="dense").astype(int)
    n_blocks = df["block"].nunique()
    log.info(
        f"{len(df)} pairs, {n_blocks} blocks, {int(df['is_self'].sum())} self pairs"
    )
    assert len(df) == len(WAVETABLES) * len(MOD_SIG_REFERENCES) * 5, len(df)
    return df.sort_values(["block", "amount"]).reset_index(drop=True)


def stimulus_names(pairs: pd.DataFrame) -> List[str]:
    """Every distinct stimulus referenced by the pair table."""
    return sorted(set(pairs["ref_stim"]) | set(pairs["var_stim"]))


# ======================================================================================
# Part 2 -- The listening test: placeholder data, screening and the noise ceiling
# ======================================================================================


def block_sensitivity(blocks: Sequence[str], seed: int = HUMAN_SEED) -> Dict[str, float]:
    """Per-block perceptual sensitivity, shared by the rating simulator and the
    positive-control embedding.

    Both have to be driven by the SAME latent or the "positive control" is not one: an
    embedding encoding a different random sensitivity than the ratings is only weakly
    related to them, and a probe failing on it would tell us nothing about the probe.
    """
    uniq = sorted(set(blocks))
    rng = np.random.default_rng(seed + 1234)
    return dict(zip(uniq, rng.uniform(0.5, 1.5, len(uniq))))


def simulate_human_ratings(pairs: pd.DataFrame, mode: str, seed: int) -> pd.DataFrame:
    """Stand-in listening test data in the same tidy shape as a real export.

    mode="noise":     ratings are independent of the stimulus. Per-participant gain and
                      bias are still applied so the file has realistic participant
                      structure, but there is no stimulus signal, so every statistic
                      must land at chance and the noise ceiling must collapse to ~0.
    mode="simulated": ratings saturate with modulation amount, with a per-block
                      sensitivity, plus participant gain/bias, trial noise and lapses.
                      A positive control for the pipeline, not a result.
    """
    rng = np.random.default_rng(seed)
    n_pairs = len(pairs)
    # Participants differ in how they use a 0-100 scale: a multiplicative gain and an
    # additive bias. Rank statistics are immune to this; it is here so the screening and
    # per-participant z-scoring code paths are actually exercised.
    gain = rng.normal(1.0, 0.25, N_PARTICIPANTS)
    bias = rng.normal(0.0, 6.0, N_PARTICIPANTS)

    if mode == "noise":
        latent = np.full(n_pairs, 50.0)  # No dependence on the stimulus whatsoever
        trial_sd = 25.0
    elif mode == "simulated":
        # Saturating in the amount rank (0 for the self pair), scaled by a sensitivity
        # that differs per block so there is genuine between-block variance to find.
        blocks = pairs["block"].to_numpy()
        sens = block_sensitivity(blocks, seed)
        steps = pairs["amount_rank"].to_numpy() - 1
        latent = np.array(
            [100.0 * sens[b] * (1.0 - np.exp(-s / 1.6)) for b, s in zip(blocks, steps)]
        )
        trial_sd = 12.0
    else:
        raise ValueError(f"Unknown HUMAN_MODE {mode}")

    ratings = latent[None, :] * gain[:, None] + bias[:, None]
    ratings = ratings + rng.normal(0.0, trial_sd, (N_PARTICIPANTS, n_pairs))
    # Lapses: a few trials per participant where attention slipped entirely
    lapse = rng.random((N_PARTICIPANTS, n_pairs)) < 0.02
    ratings[lapse] = rng.uniform(RATING_MIN, RATING_MAX, int(lapse.sum()))
    ratings = np.clip(ratings, RATING_MIN, RATING_MAX)

    long = pd.DataFrame(
        {
            "participant": np.repeat(np.arange(N_PARTICIPANTS), n_pairs),
            "wavetable": np.tile(pairs["wavetable"].to_numpy(), N_PARTICIPANTS),
            "mod_type": np.tile(pairs["mod_type"].to_numpy(), N_PARTICIPANTS),
            "amount": np.tile(pairs["amount"].to_numpy(), N_PARTICIPANTS),
            "rating": ratings.reshape(-1),
        }
    )
    log.warning(
        f"USING PLACEHOLDER LISTENING TEST DATA (HUMAN_MODE={mode!r}). "
        f"These numbers are a pipeline check, not a result."
    )
    return long


def load_human_ratings(path: str) -> pd.DataFrame:
    """Real export: columns participant, wavetable, mod_type, amount, rating."""
    long = pd.read_csv(path)
    needed = {"participant", "wavetable", "mod_type", "amount", "rating"}
    missing = needed - set(long.columns)
    assert not missing, f"{path} is missing columns {sorted(missing)}"
    log.info(f"Loaded {len(long)} ratings from {path}")
    return long


def ratings_to_matrix(long: pd.DataFrame, pairs: pd.DataFrame) -> Tuple[np.ndarray, List]:
    """Reshape tidy ratings into a (n_participants, n_pairs) matrix aligned to `pairs`.

    Keeping the full matrix rather than collapsing to means immediately is what lets us
    bootstrap over participants and estimate the noise ceiling later.
    """
    # Join on the three columns that identify a stimulus, so a real export does not have
    # to know anything about our pair_id convention.
    key = ["wavetable", "mod_type", "amount"]
    idx = pairs.reset_index()[key + ["index"]]
    merged = long.merge(idx, on=key, how="left", validate="many_to_one")
    assert merged["index"].notna().all(), "Some ratings did not match a known stimulus"
    participants = sorted(merged["participant"].unique())
    p_idx = {p: i for i, p in enumerate(participants)}

    mat = np.full((len(participants), len(pairs)), np.nan)
    mat[
        merged["participant"].map(p_idx).to_numpy(),
        merged["index"].to_numpy().astype(int),
    ] = merged["rating"].to_numpy()
    n_missing = int(np.isnan(mat).sum())
    if n_missing:
        log.warning(f"{n_missing} missing ratings; they are ignored by nanmean")
    log.info(f"Ratings matrix: {mat.shape[0]} participants x {mat.shape[1]} pairs")
    return mat, participants


def screen_participants(
    mat: np.ndarray, pairs: pd.DataFrame, participants: Sequence
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Flag participants who were not doing the task, using two checks.

    1. Hidden references. The 18 self pairs are the reference against itself, so the
       correct answer is 0. A high median there means the participant was guessing.
    2. Monotonicity. Within each block, ratings should rise with modulation amount. A
       within-block rank correlation at or below zero, pooled over blocks, is a fail.

    Returns the per-participant flag table and a boolean keep mask. Whether flagged
    participants are actually dropped is SCREEN_ACTION -- defaulting to "report",
    because a screen that silently deletes half your data is worse than no screen.
    """
    is_self = pairs["is_self"].to_numpy()
    blocks = pairs["block"].to_numpy()
    rows = []
    for i, p in enumerate(participants):
        hidden_ref_median = float(np.nanmedian(mat[i, is_self]))
        # Pool the within-block amount correlation via Fisher-z so one noisy block does
        # not dominate; 4-5 points per block is far too few to judge a block alone.
        zs = []
        for b in np.unique(blocks):
            m = (blocks == b) & ~is_self
            r = stats.spearmanr(mat[i, m], pairs["amount_rank"].to_numpy()[m]).statistic
            if np.isfinite(r):
                zs.append(np.arctanh(np.clip(r, -0.999, 0.999)))
        amount_rho = float(np.tanh(np.mean(zs))) if zs else np.nan
        rows.append(
            {
                "participant": p,
                "hidden_ref_median": hidden_ref_median,
                "amount_rho": amount_rho,
                "flag_hidden_ref": hidden_ref_median > SCREEN_MAX_HIDDEN_REF,
                "flag_monotonic": not (amount_rho > SCREEN_MIN_AMOUNT_RHO),
            }
        )
    flags = pd.DataFrame(rows)
    flags["flagged"] = flags["flag_hidden_ref"] | flags["flag_monotonic"]
    n_flagged = int(flags["flagged"].sum())
    log.info(
        f"Screening: {n_flagged}/{len(flags)} participants flagged "
        f"({int(flags['flag_hidden_ref'].sum())} on hidden references, "
        f"{int(flags['flag_monotonic'].sum())} on monotonicity)"
    )

    keep = np.ones(len(flags), dtype=bool)
    if SCREEN_ACTION == "drop":
        drop_frac = n_flagged / len(flags)
        if drop_frac > SCREEN_MAX_DROP_FRAC:
            log.warning(
                f"Refusing to drop {drop_frac:.0%} of participants (limit "
                f"{SCREEN_MAX_DROP_FRAC:.0%}). Keeping everyone; inspect the flags. "
                f"With HUMAN_MODE='noise' this is the expected outcome."
            )
        else:
            keep = ~flags["flagged"].to_numpy()
            log.info(f"Dropped {n_flagged} participants; {keep.sum()} remain")
    return flags, keep


def apply_scaling(mat: np.ndarray, mode: str) -> np.ndarray:
    """Per-participant scale correction.

    "raw" is primary: the headline statistics are rank-based, so they already ignore how
    an individual used the scale, and the raw mean is the quantity a reader expects.
    "zscore" removes per-participant gain and bias before averaging, and is reported in
    the sensitivity table -- if the two disagree, that is worth saying in the paper.

    Called on the participant's FULL response vector (all 90 trials) before the analysis
    subset is taken, because gain and bias are best estimated from everything they
    answered -- including the 18 hidden references, which are what anchor the bottom of
    the scale. Estimating them from the 72 analysed pairs alone discards that anchor.
    """
    if mode == "raw":
        return mat
    if mode == "zscore":
        mu = np.nanmean(mat, axis=1, keepdims=True)
        sd = np.nanstd(mat, axis=1, keepdims=True)
        return (mat - mu) / np.where(sd > 0, sd, 1.0)
    raise ValueError(f"Unknown scaling {mode}")


# ======================================================================================
# Part 3 -- Representations: embeddings, readouts and zero-shot distances
# ======================================================================================


@dataclass
class Repr:
    """One thing we can measure a distance with.

    `emb` maps a stimulus name to either a (dims,) clip-level vector or a
    (frames, dims) trajectory. `dist` maps a pair_id straight to a scalar distance, for
    representations where we only have distances (the calc_distances.py CSVs) or where a
    distance is all there is (the parameter oracle).

    Only reprs with `emb` can feed the probe -- a scalar cannot be re-weighted.
    """

    name: str
    emb: Optional[Dict[str, np.ndarray]] = None
    dist: Optional[Dict[str, float]] = None
    frame_rate: Optional[float] = None
    is_control: bool = False  # Negative/positive controls, kept out of the main claim
    note: str = ""


def _randomise_weights(module, seed: int) -> None:
    """Destroy the learned structure while keeping each tensor's scale.

    Permuting each parameter tensor's own values preserves its marginal distribution
    (so activations stay in a sane range) but removes everything training put there.
    This is the control that separates "the learned representation is perceptually
    aligned" from "any deep filterbank of this shape would score this well".
    """
    import torch as tr

    gen = tr.Generator().manual_seed(seed)
    with tr.no_grad():
        for p in module.parameters():
            flat = p.reshape(-1)
            p.copy_(flat[tr.randperm(flat.numel(), generator=gen)].reshape(p.shape))


def _model_factory(name: str):
    """Build one of the real embedding models from code/losses.py.

    Imported lazily and inside a try, because CLAP / PANNs / kymatio are heavy and the
    checkpoints may not be present. A missing model is skipped, never fatal.
    """
    base = name.replace("__randinit", "")
    from losses import ClapEmbeddingLoss, PANNsEmbeddingLoss

    if base == "clap_2023":
        loss_fn = ClapEmbeddingLoss(use_cuda=False, in_sr=SR)
        inner = loss_fn.model.clap.audio_encoder  # CLAP itself is not an nn.Module
    elif base.startswith("panns_"):
        variant = {
            "panns_cnn14_32k": "cnn14-32k",
            "panns_wavegram_logmel": "wavegram-logmel",
        }[base]
        loss_fn = PANNsEmbeddingLoss(variant=variant, in_sr=SR)
        inner = loss_fn.model
    else:
        raise KeyError(base)
    if name.endswith("__randinit"):
        _randomise_weights(inner, seed=ANALYSIS_SEED)
    return loss_fn


def extract_embeddings(name: str, stims: Sequence[str]) -> None:
    """Run one model over every stimulus and cache the result.

    Cached per stimulus so that adding a wavetable later does not invalidate the rest,
    and so the statistics can be iterated on without touching a network. Note the model
    sees exactly the audio the listeners heard, including CLAP's repeat-padding of the
    4 s clip to its 7 s input window (config_2023.yml, duration: 7) -- that padding
    introduces a modulation-phase discontinuity at 4 s and is a documented caveat, not
    something we silently work around.
    """
    import torch as tr
    import torchaudio

    loss_fn = _model_factory(name)
    out_dir = os.path.join(EMB_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    for stim in stims:
        out_path = os.path.join(out_dir, f"{stim}.npz")
        if os.path.exists(out_path):
            continue
        audio, sr = torchaudio.load(os.path.join(SAMPLES_DIR, f"{stim}{SUFFIX}"))
        assert sr == SR, f"Expected sr={SR}, got {sr}"
        audio = audio[:1, :]  # Samples are mono duplicated across both channels
        with tr.no_grad():
            proc = loss_fn.preproc_audio(audio)
            emb = loss_fn.get_embedding(proc)
        emb = emb.squeeze(0).cpu().numpy()  # (dims,) or (frames, dims)
        # Framewise embeddings need their frame rate stored, or the modulation-spectrum
        # readout has no way to convert an FFT bin into Hz. Derived from the audio the
        # model actually saw (post-resample, post-padding), not from the 4 s source.
        frame_rate = np.nan
        if emb.ndim == 2:
            dur_s = proc.shape[-1] / loss_fn.get_model_sr()
            frame_rate = emb.shape[0] / dur_s
        np.savez(out_path, emb=emb, frame_rate=np.asarray(frame_rate))
        log.info(f"  cached {name}/{stim} {emb.shape} @ {frame_rate:.2f} Hz")


def load_embeddings(
    name: str, stims: Sequence[str]
) -> Optional[Tuple[Dict[str, np.ndarray], Optional[float]]]:
    """Read a cached embedding set and its frame rate, or None if it is incomplete.

    Incomplete means "skip this model entirely" rather than "score it on the pairs it
    happens to cover", so a half-built cache cannot produce a statistic computed over a
    different set of pairs than the rest of the table.
    """
    out_dir = os.path.join(EMB_DIR, name)
    emb, frame_rate = {}, None
    for stim in stims:
        path = os.path.join(out_dir, f"{stim}.npz")
        if not os.path.exists(path):
            return None
        with np.load(path) as data:
            emb[stim] = data["emb"]
            if frame_rate is None and "frame_rate" in data:
                fr = float(data["frame_rate"])
                frame_rate = fr if np.isfinite(fr) else None
    return emb, frame_rate


def apply_readout(
    emb: Dict[str, np.ndarray], readout: str, frame_rate: Optional[float]
) -> Optional[Dict[str, np.ndarray]]:
    """Turn a possibly-framewise embedding into one vector per stimulus.

    "mean"      clip-level pooling, i.e. what EmbeddingLoss.forward already does. This
                is the canonical readout, and it is also nearly invariant to modulation
                RATE -- so if a model does badly on the freq condition under "mean" and
                well under the readouts below, the failure is a pooling artifact rather
                than a missing representation. That distinction is the interesting result.
    "mean_std"  adds the per-dimension temporal standard deviation, the cheapest way to
                make a clip-level summary sensitive to how much the signal moves.
    "modspec"   FFT magnitude of each dimension's trajectory, restricted to
                MOD_SPEC_FMIN_HZ..MOD_SPEC_FMAX_HZ. Directly targets modulation rate.

    Returns None when the readout needs a framewise cache and only a clip-level vector
    is available, in which case the caller skips that row.
    """
    any_arr = next(iter(emb.values()))
    if readout == "mean":
        return {k: (v.mean(axis=0) if v.ndim == 2 else v) for k, v in emb.items()}
    if any_arr.ndim != 2:
        return None  # Framewise readouts need a (frames, dims) cache
    if readout == "mean_std":
        return {
            k: np.concatenate([v.mean(axis=0), v.std(axis=0)]) for k, v in emb.items()
        }
    if readout == "modspec":
        if frame_rate is None:
            return None
        out = {}
        for k, v in emb.items():
            spec = np.abs(np.fft.rfft(v - v.mean(axis=0), axis=0))
            freqs = np.fft.rfftfreq(v.shape[0], d=1.0 / frame_rate)
            band = (freqs >= MOD_SPEC_FMIN_HZ) & (freqs <= MOD_SPEC_FMAX_HZ)
            out[k] = spec[band].reshape(-1)
        return out
    raise ValueError(f"Unknown readout {readout}")


def embedding_distances(
    emb: Dict[str, np.ndarray], pairs: pd.DataFrame, metric: str
) -> np.ndarray:
    """Zero-shot distance per pair. No fitting of any kind happens here.

    "l2"          the canonical choice: exactly what EmbeddingLoss.forward computes, so
                  a good score here means the embedding is usable as a loss as-is.
    "cosine"      scale-invariant alternative.
    "whitened_l2" per-dimension z-scoring using statistics over ALL stimuli before the
                  L2. This tests "is the default metric the problem" without fitting
                  anything to the human data, so unlike the probe it needs no CV.
    """
    mat = np.stack([emb[s] for s in stimulus_names(pairs)])
    if metric == "whitened_l2":
        mu, sd = mat.mean(axis=0), mat.std(axis=0)
        emb = {k: (v - mu) / np.where(sd > 0, sd, 1.0) for k, v in emb.items()}
        metric = "l2"
    out = []
    for ref, var in zip(pairs["ref_stim"], pairs["var_stim"]):
        a, b = emb[ref], emb[var]
        if metric == "l2":
            out.append(float(np.linalg.norm(a - b)))
        elif metric == "cosine":
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            out.append(float(1.0 - (a @ b) / denom) if denom > 0 else 0.0)
        else:
            raise ValueError(f"Unknown metric {metric}")
    return np.asarray(out)


def load_distance_csvs(pairs: pd.DataFrame) -> List[Repr]:
    """Pick up the distances calc_distances.py already computed (mse, mss, jtfs, ...).

    Reconstructs our pair_id from the CSV's own columns, and only keeps loss functions
    that cover every pair, so a partially-computed CSV cannot quietly produce a
    statistic over a different set of pairs than the other rows in the table.
    """
    reprs = []
    for fname in DISTANCE_CSVS:
        path = os.path.join(DIST_DIR, fname)
        if not os.path.exists(path):
            log.info(f"No {fname}; skipping")
            continue
        df = pd.read_csv(path)
        df["pair_id"] = (
            df["wavetable"] + "__" + df["reference"] + "__VS__" + df["mod_sig"]
        )
        for loss_name, sub in df.groupby("loss_fn"):
            dist = dict(zip(sub["pair_id"], sub["distance"]))
            missing = set(pairs["pair_id"]) - set(dist)
            if missing:
                log.warning(
                    f"{fname}:{loss_name} covers {len(pairs) - len(missing)}/"
                    f"{len(pairs)} pairs; skipping"
                )
                continue
            reprs.append(Repr(name=str(loss_name), dist=dist, note=fname))
    return reprs


def build_reprs(pairs: pd.DataFrame) -> List[Repr]:
    """Assemble every representation to be scored, in table order.

    Always available (so the script runs with no checkpoints at all):
      amount_rank          the parameter oracle. A representation that cannot match this
                           has added nothing beyond "more modulation sounds different".
      noise_768            a Gaussian embedding containing nothing. Its scores are what
                           chance looks like for a 768-dim embedding under each
                           statistic, which is the point: chance is not zero.
      synthetic_signal     an embedding that genuinely encodes the manipulation in 3 of
                           768 dims. A positive control for the pipeline's power; only
                           meaningful with HUMAN_MODE="simulated".
    """
    stims = stimulus_names(pairs)
    rng = np.random.default_rng(ANALYSIS_SEED)
    reprs: List[Repr] = []

    # --- Parameter oracle: distance == ordinal position of the amount in its block ----
    reprs.append(
        Repr(
            name="amount_rank",
            dist=dict(zip(pairs["pair_id"], (pairs["amount_rank"] - 1).astype(float))),
            note="parameter oracle",
        )
    )

    # --- Synthetic controls ----------------------------------------------------------
    n_dims = 768
    noise_emb = {s: rng.normal(0.0, 1.0, n_dims) for s in stims}
    reprs.append(
        Repr(name="noise_768", emb=noise_emb, is_control=True, note="negative control")
    )

    # Same noise, but 3 dimensions carry the manipulation: overall amount, amount scaled
    # by a per-wavetable sensitivity, and wavetable identity. Buried in 765 noise dims,
    # which is the regime a real embedding is in.
    steps = dict(zip(pairs["var_stim"], (pairs["amount_rank"] - 1).astype(float)))
    wt_of = dict(zip(pairs["var_stim"], pairs["wavetable"]))
    block_of = dict(zip(pairs["var_stim"], pairs["block"]))
    sens = block_sensitivity(pairs["block"].to_numpy(), HUMAN_SEED)
    sig_emb = {}
    for s in stims:
        v = rng.normal(0.0, 1.0, n_dims).copy()
        step = steps.get(s, 0.0)
        # Dim 0 carries exactly the latent the simulated ratings are built from, dim 1
        # the raw amount, dim 2 wavetable identity -- buried in 765 noise dims, which is
        # the regime a real embedding is in.
        v[0] += 2.0 * step * sens.get(block_of.get(s, ""), 1.0)
        v[1] += 1.2 * step
        v[2] += 0.8 * WAVETABLES.index(wt_of.get(s, WAVETABLES[0]))
        sig_emb[s] = v
    reprs.append(
        Repr(
            name="synthetic_signal",
            emb=sig_emb,
            is_control=True,
            note="positive control (only meaningful with HUMAN_MODE='simulated')",
        )
    )

    # --- Real models -----------------------------------------------------------------
    for name in EMBEDDING_MODELS:
        loaded = load_embeddings(name, stims)
        if loaded is None and EXTRACT_MISSING_EMBEDDINGS:
            try:
                extract_embeddings(name, stims)
                loaded = load_embeddings(name, stims)
            except Exception as e:  # noqa: BLE001 - a missing checkpoint is not fatal
                log.warning(f"Could not extract {name}: {type(e).__name__}: {e}")
        if loaded is None:
            log.warning(
                f"No cached embeddings for {name} in {os.path.join(EMB_DIR, name)}; "
                f"skipping (set EXTRACT_MISSING_EMBEDDINGS=True to build the cache)"
            )
            continue
        emb, frame_rate = loaded
        reprs.append(
            Repr(
                name=name,
                emb=emb,
                frame_rate=frame_rate,
                is_control=name.endswith("__randinit"),
            )
        )

    # --- Distances already on disk ---------------------------------------------------
    reprs.extend(load_distance_csvs(pairs))
    log.info(f"{len(reprs)} representations: {[r.name for r in reprs]}")
    return reprs


# ======================================================================================
# Part 4 -- The zero-shot statistics
#
# Three statistics, because a single correlation conflates three different things. Within
# a block both the ratings and almost any distance rise with modulation amount, so a
# pooled correlation reads high for plain MSE and that is not evidence of perceptual
# content. Each statistic isolates one component.
# ======================================================================================


@dataclass
class StatContext:
    """Everything a statistic needs about the pair table, precomputed once."""

    blocks: np.ndarray
    wavetables: np.ndarray
    mod_types: np.ndarray
    # Index pairs (i, j) of comparisons that live in the SAME block, for the
    # block-restricted Kendall tau. Precomputed because it is inside every permutation.
    cmp_i: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    cmp_j: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    @staticmethod
    def build(pairs: pd.DataFrame) -> "StatContext":
        blocks = pairs["block"].to_numpy()
        ii, jj = [], []
        for b in np.unique(blocks):
            idx = np.where(blocks == b)[0]
            for a in range(len(idx)):
                for c in range(a + 1, len(idx)):
                    ii.append(idx[a])
                    jj.append(idx[c])
        return StatContext(
            blocks=blocks,
            wavetables=pairs["wavetable"].to_numpy(),
            mod_types=pairs["mod_type"].to_numpy(),
            cmp_i=np.asarray(ii, dtype=int),
            cmp_j=np.asarray(jj, dtype=int),
        )


def block_restricted_kendall(d: np.ndarray, y: np.ndarray, ctx: StatContext) -> float:
    """Statistic 1: does the model order pairs correctly WITHIN a block?

    Kendall's tau over only those comparisons whose two pairs share a block (18 blocks x
    C(4,2) = 108 comparisons). This is the well-powered version of "per-block Spearman",
    which on 4 points per block is far too noisy to report. Being block-restricted, it
    is blind to the between-block main effect that inflates the pooled correlation.
    """
    sgn = np.sign(d[ctx.cmp_i] - d[ctx.cmp_j]) * np.sign(y[ctx.cmp_i] - y[ctx.cmp_j])
    n_conc, n_disc = int((sgn > 0).sum()), int((sgn < 0).sum())
    if n_conc + n_disc == 0:
        return np.nan
    return (n_conc - n_disc) / (n_conc + n_disc)


def cross_block_spearman(d: np.ndarray, y: np.ndarray, ctx: StatContext) -> float:
    """Statistic 2: is the model CALIBRATED across blocks?

    Collapse each block to its mean and correlate the 18 block-level values. This asks
    whether the model knows that a given depth on one wavetable sounds more different
    than the same depth on another, and that rate changes are less salient than depth
    changes. Almost everything fails here, and with only 6 wavetables the CI is wide --
    which is exactly why it gets its own row instead of being buried in a pooled number.
    """
    uniq = np.unique(ctx.blocks)
    # NaN here is meaningful, not a failure: a representation whose block means are all
    # identical (the amount_rank oracle, by construction) carries no cross-block
    # information at all, so the statistic is undefined rather than zero.
    dm = np.array([d[ctx.blocks == b].mean() for b in uniq])
    ym = np.array([y[ctx.blocks == b].mean() for b in uniq])
    return float(stats.spearmanr(dm, ym).statistic)


def pooled_spearman(d: np.ndarray, y: np.ndarray, ctx: StatContext) -> float:
    """Statistic 3: the pooled correlation, reported LAST and with a caveat.

    Contains statistics 1 and 2 plus the trivial "both quantities increase with the
    parameter we swept" main effect. It is here because readers expect it, not because
    it is the most informative number.
    """
    return float(stats.spearmanr(d, y).statistic)


def mod_type_spearman(
    d: np.ndarray, y: np.ndarray, ctx: StatContext, mod_type: str
) -> float:
    """Pooled Spearman restricted to one modulation type (24 pairs).

    The per-condition breakdown is where the story usually is: clip-level pooling
    should track amp depth reasonably and be close to blind to freq rate.
    """
    m = ctx.mod_types == mod_type
    if m.sum() < 3:
        return np.nan
    return float(stats.spearmanr(d[m], y[m]).statistic)


def isotonic_r2(d: np.ndarray, y: np.ndarray, ctx: StatContext) -> float:
    """Variance explained under the best MONOTONE mapping, cross-validated.

    Human ratings are bounded 0-100 and saturate; model distances are unbounded and
    roughly linear in the parameter. Pearson r on the raw values would punish that
    mismatch, so fit an isotonic (monotone, non-parametric) link instead. Held out by
    wavetable so the link cannot be fitted to the points it is scored on.
    """
    pred = np.full(len(y), np.nan)
    for wt in np.unique(ctx.wavetables):
        te = ctx.wavetables == wt
        tr_ = ~te
        if tr_.sum() < 3 or te.sum() == 0:
            continue
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(d[tr_], y[tr_])
        pred[te] = iso.predict(d[te])
    ok = np.isfinite(pred)
    if ok.sum() < 3:
        return np.nan
    sse = float(np.sum((pred[ok] - y[ok]) ** 2))
    sst = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    return 1.0 - sse / sst if sst > 0 else np.nan


def reliable_agreement(
    d: np.ndarray, mat: np.ndarray, ctx: StatContext, alpha: float = 0.05
) -> Tuple[float, int, int]:
    """How often does the model agree with listeners, where listeners agree with each other?

    For every within-block comparison, test whether the 50 participants reliably rated
    one stimulus as more different than the other (paired Wilcoxon signed-rank, BH
    corrected). On the surviving subset, report the fraction where the model's distance
    ordering matches. This is the interpretable companion to tau: "CLAP agrees with
    listeners on 78% of the comparisons listeners agree on".
    """
    pvals, signs = [], []
    for i, j in zip(ctx.cmp_i, ctx.cmp_j):
        diff = mat[:, i] - mat[:, j]
        diff = diff[np.isfinite(diff)]
        if len(diff) < 5 or np.allclose(diff, 0.0):
            pvals.append(1.0)
            signs.append(0.0)
            continue
        try:
            p = float(stats.wilcoxon(diff, zero_method="zsplit").pvalue)
        except ValueError:  # All-zero differences
            p = 1.0
        pvals.append(p)
        signs.append(np.sign(np.mean(diff)))
    keep = benjamini_hochberg(np.asarray(pvals), alpha)
    n_reliable = int(keep.sum())
    if n_reliable == 0:
        # Expected with HUMAN_MODE="noise": listeners never reliably agree, so there is
        # nothing to agree with.
        return np.nan, 0, len(pvals)
    model_sign = np.sign(d[ctx.cmp_i] - d[ctx.cmp_j])
    agree = model_sign[keep] == np.asarray(signs)[keep]
    return float(agree.mean()), n_reliable, len(pvals)


def benjamini_hochberg(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """BH step-up. Returns a boolean mask of hypotheses declared significant."""
    n = len(pvals)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = pvals[order] <= thresh
    keep = np.zeros(n, dtype=bool)
    if passed.any():
        k = np.max(np.where(passed)[0])
        keep[order[: k + 1]] = True
    return keep


# ======================================================================================
# Part 5 -- Inference: the noise ceiling, permutation nulls and cluster bootstraps
#
# The three things that make a number in Part 4 interpretable:
#   ceiling      how high could a PERFECT model score, given how noisy 50 listeners are?
#   null         what does this statistic read when there is nothing to find? Measured by
#                re-running the pipeline on shuffled targets, because for these
#                statistics chance is provably not zero.
#   CI           how much would the number move with different wavetables / listeners?
# ======================================================================================

# name -> (function, which permutation scheme tests it)
StatFn = Callable[[np.ndarray, np.ndarray, StatContext], float]
STATS: Dict[str, Tuple[StatFn, str]] = {
    "tau_within_block": (block_restricted_kendall, "within_block"),
    "rho_cross_block": (cross_block_spearman, "block_label"),
    "rho_pooled": (pooled_spearman, "within_block"),
    "rho_amp": (lambda d, y, c: mod_type_spearman(d, y, c, "amp"), "within_block"),
    "rho_freq": (lambda d, y, c: mod_type_spearman(d, y, c, "freq"), "within_block"),
    "rho_reg": (lambda d, y, c: mod_type_spearman(d, y, c, "reg"), "within_block"),
    "r2_isotonic": (isotonic_r2, "within_block"),
}
# The statistic quoted in the abstract. Within-block ordering is the claim that a
# distance function actually has to get right to be usable as a perceptual loss.
HEADLINE_STAT = "tau_within_block"


def permute_within_blocks(y: np.ndarray, ctx: StatContext, rng) -> np.ndarray:
    """Shuffle the human values WITHIN each block.

    This is exactly what the null "the model's within-block ordering is chance" asserts
    is exchangeable. It leaves every block mean untouched, so a pooled statistic tested
    against this null is being tested on its within-block component only -- which is the
    honest interpretation, and worth stating in the paper.
    """
    out = y.copy()
    for b in np.unique(ctx.blocks):
        m = ctx.blocks == b
        out[m] = y[m][rng.permutation(int(m.sum()))]
    return out


def permute_block_labels(y: np.ndarray, ctx: StatContext, rng) -> np.ndarray:
    """Swap whole blocks of human values between blocks.

    Preserves the within-block pattern and destroys the block-to-block correspondence,
    which is the correct null for the cross-block statistic. Within-block shuffling
    would leave that statistic completely unchanged.
    """
    uniq = np.unique(ctx.blocks)
    perm = rng.permutation(len(uniq))
    out = y.copy()
    for src, dst in zip(uniq, uniq[perm]):
        m_src, m_dst = ctx.blocks == src, ctx.blocks == dst
        if m_src.sum() == m_dst.sum():
            out[m_dst] = y[m_src]
    return out


def split_half_ceiling(
    stat_fn: StatFn,
    mat: np.ndarray,
    ctx: StatContext,
    n_splits: int,
    rng,
    is_correlation: bool,
) -> float:
    """Noise ceiling for an arbitrary statistic, in that statistic's own units.

    Split the participants in half, then feed one half's mean ratings in as the
    "distance" and the other half's as the target. Whatever the statistic reads is what
    listeners themselves achieve against listeners -- no model can do better. Doing it
    per statistic avoids the mistake of dividing a Kendall tau by a Spearman ceiling.

    This is the CONSERVATIVE ceiling: each half has only n/2 participants, so it
    understates what is achievable against the full 50-participant mean. See
    spearman_brown_upper for the optimistic counterpart.
    """
    n_part = mat.shape[0]
    vals = []
    for _ in range(n_splits):
        perm = rng.permutation(n_part)
        a, b = perm[: n_part // 2], perm[n_part // 2 :]
        v = stat_fn(np.nanmean(mat[a], axis=0), np.nanmean(mat[b], axis=0), ctx)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan
    vals = np.asarray(vals)
    # Correlations are averaged in Fisher-z space, because the sampling distribution of a
    # correlation is skewed near +-1. R2 is not a correlation and must not be transformed
    # that way, so the caller states which kind of statistic this is rather than letting
    # the code guess from the value range (an R2 that happens to land inside [-1, 1]
    # would otherwise be silently mis-averaged).
    if is_correlation:
        return float(np.tanh(np.mean(np.arctanh(np.clip(vals, -0.999, 0.999)))))
    return float(np.mean(vals))


def spearman_brown_upper(r_half: float) -> float:
    """Optimistic ceiling for a Spearman-type statistic against the 50-listener mean.

    Spearman-Brown lifts the half-sample reliability r_half to the reliability of the
    full mean, 2r/(1+r); the square root of that is the highest correlation a noiseless
    model could have with the observed mean. Note this is derived for Pearson and is
    being applied to a rank correlation, so it is an approximation -- say so in the paper
    and report the conservative split-half value next to it.
    """
    if not np.isfinite(r_half) or r_half <= 0:
        return np.nan
    return float(np.sqrt(2.0 * r_half / (1.0 + r_half)))


def resample_wavetables(pairs: pd.DataFrame, rng) -> Tuple[np.ndarray, StatContext]:
    """One cluster-bootstrap draw over the 6 wavetables.

    Wavetables are the honest resampling unit: pairs inside a wavetable are not
    independent. With only 6 clusters the intervals come out wide, which is the correct
    representation of what this design can support, not a defect to be tuned away.

    The two label columns are deliberately treated differently, and getting this wrong
    silently inflates a statistic:
      block      gets a per-copy suffix, so a wavetable drawn twice contributes two
                 separate blocks rather than one doubled block. Without this, the
                 block-restricted tau would form comparisons between two copies of the
                 same pair, which are identical by construction.
      wavetable  keeps its ORIGINAL name, so every copy of a wavetable lands in the same
                 cross-validation fold. Without this, isotonic_r2's leave-one-wavetable-
                 out CV trains on one copy and tests on another copy of the same
                 wavetable -- pure leakage, which shifts that statistic's CI upward.
    """
    picks = rng.choice(WAVETABLES, size=len(WAVETABLES), replace=True)
    idx, blocks, wts, mods = [], [], [], []
    for copy_i, wt in enumerate(picks):
        m = np.where(pairs["wavetable"].to_numpy() == wt)[0]
        idx.extend(m.tolist())
        blocks.extend([f"{b}__c{copy_i}" for b in pairs["block"].to_numpy()[m]])
        wts.extend([wt] * len(m))
        mods.extend(pairs["mod_type"].to_numpy()[m].tolist())
    idx = np.asarray(idx, dtype=int)
    sub = pd.DataFrame(
        {"block": blocks, "wavetable": wts, "mod_type": mods, "amount_rank": 0}
    )
    return idx, StatContext.build(sub)


def bootstrap_ci(vals: Sequence[float], q: float = 95.0) -> Tuple[float, float]:
    """Percentile CI, ignoring draws where the statistic was undefined."""
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if len(v) < 10:
        return np.nan, np.nan
    lo, hi = (100.0 - q) / 2.0, 100.0 - (100.0 - q) / 2.0
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def permutation_p(observed: float, null: Sequence[float]) -> float:
    """One-sided p: how often does chance reach the observed value?

    The +1s are the standard finite-sample correction, so p is never exactly 0.
    """
    v = np.asarray([x for x in null if np.isfinite(x)])
    if not np.isfinite(observed) or len(v) == 0:
        return np.nan
    return float((np.sum(v >= observed) + 1) / (len(v) + 1))


# ======================================================================================
# Part 6 -- The bounded probe
#
# "The information is in the embedding, plain L2 just does not expose it." Secondary by
# construction, with four guardrails, because with 72 points and 768 dims an
# unconstrained fit is unfalsifiable:
#   1. ridge only, never an MLP -- an MLP has more capacity than this dataset has
#      information and would fit 72 points of pure noise
#   2. unsupervised PCA to PROBE_PCA_DIMS, fitted INSIDE the training fold on the
#      embeddings only, so no label information can leak through it -- reported
#      alongside the no-PCA variant, because PCA keeps the highest-variance directions
#      and can throw the signal away when it lives in a few dimensions of an
#      otherwise isotropic feature set (measured: on a synthetic embedding with the
#      signal in 3 of 768 dims, PCA-16 misses it at p=0.12 while all-dims finds it at
#      p=0.04)
#   3. outer CV holds out whole wavetables or whole mod types; alpha is chosen by an
#      inner CV on the training fold only
#   4. the permutation null re-runs the ENTIRE procedure (alpha selection included) on
#      shuffled targets, and its mean is reported, because chance here is not 0
# ======================================================================================


def probe_features(emb: Dict[str, np.ndarray], pairs: pd.DataFrame) -> np.ndarray:
    """Pair features: the elementwise |difference| of the two stimulus embeddings.

    The human answer is a dissimilarity, so the feature has to be a function of the
    PAIR. Using the variant's embedding alone would let the probe identify which stimulus
    it is and read off the amount. The absolute difference also enforces the symmetry a
    distance must have, so a fitted weighting stays usable as a loss.
    """
    return np.stack(
        [
            np.abs(emb[ref] - emb[var])
            for ref, var in zip(pairs["ref_stim"], pairs["var_stim"])
        ]
    )


@dataclass
class ProbeFold:
    """One outer fold, with everything that does not depend on the targets precomputed.

    The ridge is solved in DUAL form: there are ~60 training pairs and up to a few
    thousand feature dimensions, so working in the n x n Gram matrix instead of the
    k x k covariance makes the cost independent of the number of dimensions. The
    eigendecomposition of the Gram matrix is taken once here, which turns the entire
    alpha search into a few matrix-vector products -- necessary because alpha selection
    has to be re-run inside every permutation.
    """

    tr: np.ndarray  # Training mask
    te: np.ndarray  # Test mask
    u: np.ndarray  # Eigenvectors of the training Gram matrix
    lam: np.ndarray  # Eigenvalues, non-negative
    k_te: np.ndarray  # Test-vs-train Gram block


def _ridge_loo_alpha(fold: ProbeFold, y_tr: np.ndarray, alphas: np.ndarray) -> float:
    """Pick alpha by exact leave-one-out on the TRAINING fold only.

    Exact because ridge LOO has a closed form via the hat matrix diagonal, so no refits
    are needed: loo_residual_i = residual_i / (1 - h_ii). With the eigendecomposition in
    hand, h_ii and the residuals for a given alpha are O(n^2).

    Note this inner LOO is over training *stimuli*, which is fine: it only selects a
    hyperparameter, and the number it produces is never scored. The outer split is what
    has to respect the cluster structure.
    """
    n = len(y_tr)
    yc = y_tr - y_tr.mean()
    uty = fold.u.T @ yc
    u_sq = fold.u ** 2
    best, best_mse = alphas[0], np.inf
    for a in alphas:
        shrink = fold.lam / (fold.lam + a)  # H = U diag(shrink) U^T
        yhat = fold.u @ (shrink * uty)
        h = u_sq @ shrink + 1.0 / n  # +1/n for the intercept
        resid = yc - yhat
        mse = float(np.mean((resid / np.clip(1.0 - h, 1e-9, None)) ** 2))
        if mse < best_mse:
            best, best_mse = a, mse
    return best


def _prepare_folds(X: np.ndarray, groups: np.ndarray, k: Optional[int]) -> List[ProbeFold]:
    """Standardise, optionally PCA-project, and factorise once per fold.

    None of these steps uses the targets, so caching them across permutations is not
    leakage -- and it is what makes a few hundred permutations of the full probe
    affordable. k=None skips the PCA and keeps every standardised dimension.
    """
    folds = []
    for g in np.unique(groups):
        te = groups == g
        tr_ = ~te
        if tr_.sum() < 5 or te.sum() == 0:
            continue
        # Standardisation statistics come from the training fold only
        mu, sd = X[tr_].mean(axis=0), X[tr_].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Xtr, Xte = (X[tr_] - mu) / sd, (X[te] - mu) / sd
        centre = Xtr.mean(axis=0)
        Xtr, Xte = Xtr - centre, Xte - centre
        if k is not None:
            _, _, Vt = np.linalg.svd(Xtr, full_matrices=False)
            n_comp = int(min(k, Vt.shape[0], tr_.sum() - 2))
            P = Vt[:n_comp].T
            Xtr, Xte = Xtr @ P, Xte @ P
        gram = Xtr @ Xtr.T
        lam, u = np.linalg.eigh(gram)
        folds.append(
            ProbeFold(tr=tr_, te=te, u=u, lam=np.clip(lam, 0.0, None), k_te=Xte @ Xtr.T)
        )
    return folds


def probe_cv(folds: List[ProbeFold], y: np.ndarray, ctx: StatContext) -> Tuple[float, float]:
    """Cross-validated ridge probe. Returns (Spearman, R2) on held-out predictions.

    Spearman is the primary metric rather than R2 because the human target is a bounded,
    saturating 0-100 scale: a probe can order the pairs perfectly and still miss on R2
    purely because of the link function.
    """
    pred = np.full(len(y), np.nan)
    for f in folds:
        y_tr = y[f.tr]
        alpha = _ridge_loo_alpha(f, y_tr, RIDGE_ALPHAS)
        mu_y = y_tr.mean()
        # Dual solution: w_dual = U diag(1/(lam + a)) U^T y_c, prediction = K_te w_dual
        dual = f.u @ ((f.u.T @ (y_tr - mu_y)) / (f.lam + alpha))
        pred[f.te] = f.k_te @ dual + mu_y
    ok = np.isfinite(pred)
    if ok.sum() < 5:
        return np.nan, np.nan
    rho = float(stats.spearmanr(pred[ok], y[ok]).statistic)
    sse = float(np.sum((pred[ok] - y[ok]) ** 2))
    sst = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    return rho, (1.0 - sse / sst if sst > 0 else np.nan)


def run_probe(
    reprs: Sequence[Repr],
    pairs: pd.DataFrame,
    y: np.ndarray,
    ctx: StatContext,
    ceilings: Dict[str, float],
    rng,
) -> pd.DataFrame:
    """Probe every representation that has embeddings, under every outer CV scheme."""
    scheme_groups = {
        "leave_one_wavetable_out": ctx.wavetables,
        "leave_one_mod_type_out": ctx.mod_types,
    }
    rows = []
    for r in reprs:
        if r.emb is None:
            continue  # A scalar distance cannot be re-weighted
        readout = apply_readout(r.emb, CANONICAL_READOUT, r.frame_rate)
        X = probe_features(readout, pairs)
        for scheme in PROBE_SCHEMES:
            for k_dims in PROBE_DIM_VARIANTS:
                folds = _prepare_folds(X, scheme_groups[scheme], k_dims)
                rho, r2 = probe_cv(folds, y, ctx)
                # Null: the same pipeline -- PCA basis, alpha selection and all -- with the
                # targets shuffled within block. This is the only way to know where chance is.
                null = [
                    probe_cv(folds, permute_within_blocks(y, ctx, rng), ctx)[0]
                    for _ in range(N_PERM_PROBE)
                ]
                null = np.asarray([v for v in null if np.isfinite(v)])
                rows.append(
                    {
                        "repr": r.name,
                        "scheme": scheme,
                        "n_dims_in": X.shape[1],
                        "n_dims_pca": k_dims if k_dims is not None else X.shape[1],
                        "rho": rho,
                        "r2": r2,
                        "null_mean_rho": float(null.mean()) if len(null) else np.nan,
                        "null_p95_rho": float(np.percentile(null, 95)) if len(null) else np.nan,
                        "p_perm": permutation_p(rho, null),
                        "ceiling_rho_pooled": ceilings.get("rho_pooled", np.nan),
                        "is_control": r.is_control,
                    }
                )
                log.info(
                    f"  probe {r.name:<24} {scheme:<24} k={str(k_dims):<5} "
                    f"rho={rho:+.3f} null={rows[-1]['null_mean_rho']:+.3f} "
                    f"p={rows[-1]['p_perm']:.3f}"
                )
    df = pd.DataFrame(rows)
    if len(df):
        # BH within each (scheme, dimensionality): the family is "all representations,
        # this exact probe configuration".
        #
        # A significant flag also requires rho > 0. The grouped-CV null sits well below
        # zero here (predicting a held-out wavetable from uninformative features is
        # anti-correlated), so a probe that orders the pairs BACKWARDS at rho = -0.25 can
        # still clear a null of -0.38 and earn p = 0.035. That says the estimator is
        # biased, not that the embedding predicts perception. The raw p stays in the CSV.
        for _, sub in df.groupby(["scheme", "n_dims_pca"]):
            keep = benjamini_hochberg(sub["p_perm"].fillna(1.0).to_numpy(), 0.05)
            df.loc[sub.index, "significant_bh"] = keep & (sub["rho"].to_numpy() > 0)
    return df


# ======================================================================================
# Part 7 -- Running the zero-shot table
# ======================================================================================


def canonical_distance(r: Repr, pairs: pd.DataFrame) -> Optional[np.ndarray]:
    """The one distance per representation that goes in the main table.

    Deliberately fixed rather than swept: the canonical config is the one you would
    actually use as a loss (clip-level mean pooling, L2), so the main table has one row
    per representation and a stated multiplicity. Everything else is a sensitivity row.
    """
    if r.dist is not None:
        return np.asarray([r.dist[p] for p in pairs["pair_id"]], dtype=float)
    readout = apply_readout(r.emb, CANONICAL_READOUT, r.frame_rate)
    if readout is None:
        return None
    return embedding_distances(readout, pairs, CANONICAL_METRIC)


def run_zero_shot(
    reprs: Sequence[Repr],
    pairs: pd.DataFrame,
    y: np.ndarray,
    mat: np.ndarray,
    ctx: StatContext,
    ceilings: Dict[str, float],
    rng,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Score every representation on every statistic, with nulls and CIs.

    The permutation draws and the bootstrap resamples are generated ONCE and shared
    across representations. That is not just a speed trick: paired resamples are what
    make the model-vs-baseline difference CI valid.
    """
    # --- shared randomisation --------------------------------------------------------
    nulls = {
        "within_block": [permute_within_blocks(y, ctx, rng) for _ in range(N_PERMUTATIONS)],
        "block_label": [permute_block_labels(y, ctx, rng) for _ in range(N_PERMUTATIONS)],
    }
    boot_wt = [resample_wavetables(pairs, rng) for _ in range(N_BOOTSTRAPS)]
    boot_part = [
        rng.integers(0, mat.shape[0], mat.shape[0]) for _ in range(N_BOOTSTRAPS)
    ]
    boot_part_y = [np.nanmean(mat[p], axis=0) for p in boot_part]

    dists = {}
    for r in reprs:
        d = canonical_distance(r, pairs)
        if d is None:
            log.warning(f"{r.name}: canonical readout unavailable; skipping")
            continue
        dists[r.name] = d

    rows = []
    for r in reprs:
        if r.name not in dists:
            continue
        d = dists[r.name]
        d_base = dists.get(COMPARISON_BASELINE)
        agree, n_reliable, n_cmp = reliable_agreement(d, mat, ctx)
        for stat_name, (stat_fn, null_kind) in STATS.items():
            obs = stat_fn(d, y, ctx)
            null = [stat_fn(d, y_p, ctx) for y_p in nulls[null_kind]]
            null = np.asarray([v for v in null if np.isfinite(v)])
            ci_wt = bootstrap_ci(
                [stat_fn(d[i], y[i], c) for i, c in boot_wt]
            )
            ci_part = bootstrap_ci([stat_fn(d, y_b, ctx) for y_b in boot_part_y])
            # Paired difference against the "did you need a network" reference point
            if d_base is not None and r.name != COMPARISON_BASELINE:
                diffs = [
                    stat_fn(d[i], y[i], c) - stat_fn(d_base[i], y[i], c)
                    for i, c in boot_wt
                ]
                d_lo, d_hi = bootstrap_ci(diffs)
                # All-NaN happens legitimately: amount_rank has identical block means in
                # every block, so its cross-block statistic is undefined rather than bad.
                d_mean = (
                    float(np.nanmean(diffs))
                    if np.any(np.isfinite(diffs))
                    else np.nan
                )
            else:
                d_lo = d_hi = d_mean = np.nan
            ceil = ceilings.get(stat_name, np.nan)
            rows.append(
                {
                    "repr": r.name,
                    "stat": stat_name,
                    "value": obs,
                    "ci_lo_wavetable": ci_wt[0],
                    "ci_hi_wavetable": ci_wt[1],
                    "ci_lo_participant": ci_part[0],
                    "ci_hi_participant": ci_part[1],
                    "null_mean": float(null.mean()) if len(null) else np.nan,
                    "null_p95": float(np.percentile(null, 95)) if len(null) else np.nan,
                    "p_perm": permutation_p(obs, null),
                    "ceiling": ceil,
                    # Guarded: dividing by a small ceiling amplifies noise more than it
                    # corrects for it. With HUMAN_MODE="noise" this is NaN by design.
                    "value_over_ceiling": (
                        obs / ceil
                        if np.isfinite(ceil) and ceil >= MIN_USABLE_CEILING
                        else np.nan
                    ),
                    f"delta_vs_{COMPARISON_BASELINE}": d_mean,
                    "delta_ci_lo": d_lo,
                    "delta_ci_hi": d_hi,
                    "agreement_on_reliable": agree,
                    "n_reliable_comparisons": n_reliable,
                    "n_comparisons": n_cmp,
                    "is_control": r.is_control,
                    "note": r.note,
                }
            )
        log.info(
            f"  {r.name:<24} {HEADLINE_STAT}="
            f"{[x for x in rows if x['repr'] == r.name and x['stat'] == HEADLINE_STAT][0]['value']:+.3f}"
        )
    df = pd.DataFrame(rows)
    # BH within each statistic: the family is "all representations, this statistic".
    # Flagging additionally requires a positive value -- see note above run_probe.
    for stat_name, sub in df.groupby("stat"):
        keep = benjamini_hochberg(sub["p_perm"].fillna(1.0).to_numpy(), 0.05)
        df.loc[sub.index, "significant_bh"] = keep & (sub["value"].to_numpy() > 0)
    return df, dists


def run_sensitivity(
    reprs: Sequence[Repr],
    pairs_full: pd.DataFrame,
    mat_full: np.ndarray,
    keep_part: np.ndarray,
) -> pd.DataFrame:
    """Point estimates under every analysis choice we made, so none of them is hidden.

    Four axes: self-pairs in/out, raw vs per-participant z-scored ratings, the
    non-canonical distance metrics, and the framewise readouts. No permutation tests
    here on purpose -- these are robustness checks, and testing all of them would inflate
    the multiplicity of the main table. The number of variants is logged so the paper can
    state it.
    """
    rows = []
    for drop_self in [True, False]:
        sel = (~pairs_full["is_self"].to_numpy()) if drop_self else np.ones(len(pairs_full), bool)
        pairs_s = pairs_full[sel].reset_index(drop=True)
        ctx_s = StatContext.build(pairs_s)
        for scaling in ["raw", "zscore"]:
            mat_s = apply_scaling(mat_full[keep_part], scaling)[:, sel]
            y_s = np.nanmean(mat_s, axis=0)
            for r in reprs:
                variants: List[Tuple[str, Optional[np.ndarray]]] = []
                if r.dist is not None:
                    variants.append(
                        (
                            f"{CANONICAL_READOUT}/{CANONICAL_METRIC}",
                            np.asarray([r.dist[p] for p in pairs_s["pair_id"]]),
                        )
                    )
                else:
                    for readout in [CANONICAL_READOUT] + SENSITIVITY_READOUTS:
                        emb_r = apply_readout(r.emb, readout, r.frame_rate)
                        if emb_r is None:
                            continue
                        metrics = (
                            [CANONICAL_METRIC] + SENSITIVITY_METRICS
                            if readout == CANONICAL_READOUT
                            else [CANONICAL_METRIC]
                        )
                        for metric in metrics:
                            variants.append(
                                (
                                    f"{readout}/{metric}",
                                    embedding_distances(emb_r, pairs_s, metric),
                                )
                            )
                for label, d in variants:
                    if d is None:
                        continue
                    rows.append(
                        {
                            "repr": r.name,
                            "variant": label,
                            "drop_self_pairs": drop_self,
                            "scaling": scaling,
                            "tau_within_block": block_restricted_kendall(d, y_s, ctx_s),
                            "rho_cross_block": cross_block_spearman(d, y_s, ctx_s),
                            "rho_pooled": pooled_spearman(d, y_s, ctx_s),
                        }
                    )
    df = pd.DataFrame(rows)
    log.info(f"Sensitivity table: {len(df)} variants (state this count in the paper)")
    return df


# ======================================================================================
# Part 8 -- Figures
# ======================================================================================


def _style(ax) -> None:
    ax.grid(True, color=AXIS_COLOR, alpha=0.15, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


MOD_COLORS = {"amp": "#2a78d6", "freq": "#d6552a", "reg": "#3f9e5a"}


def plot_rank_scatter(
    dists: Dict[str, np.ndarray],
    y: np.ndarray,
    pairs: pd.DataFrame,
    ceilings: Dict[str, float],
    save_dir: str,
) -> None:
    """Rank-rank scatter per representation, coloured by modulation type.

    Ranks rather than raw values because the headline statistics are rank-based and the
    two quantities have incomparable units. Colouring by mod type makes the failure mode
    visible at a glance: three separated clouds means the model is only tracking the
    between-condition main effect.
    """
    names = list(dists)
    n_col = min(3, len(names))
    n_row = int(np.ceil(len(names) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.2 * n_col, 4.2 * n_row), squeeze=False)
    y_rank = stats.rankdata(y)
    for ax, name in zip(axes.flat, names):
        d_rank = stats.rankdata(dists[name])
        for mt, color in MOD_COLORS.items():
            m = pairs["mod_type"].to_numpy() == mt
            ax.scatter(d_rank[m], y_rank[m], s=28, color=color, alpha=0.85, label=mt)
        tau = block_restricted_kendall(dists[name], y, StatContext.build(pairs))
        ax.set_title(f"{name}\n{HEADLINE_STAT} = {tau:+.3f}", fontsize=9)
        ax.set_xlabel("model distance (rank)")
        ax.set_ylabel("mean listener rating (rank)")
        _style(ax)
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="best", frameon=False, fontsize=8, labelcolor=AXIS_COLOR)
    fig.suptitle(
        f"Listener dissimilarity vs model distance  "
        f"(ceiling for rho_pooled = {ceilings.get('rho_pooled', float('nan')):.2f})",
        fontsize=11,
    )
    fig.tight_layout()
    path = os.path.join(save_dir, "fig1_rank_scatter.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    log.info(f"Saved {path}")


def plot_scores(df: pd.DataFrame, ceilings: Dict[str, float], save_dir: str) -> None:
    """Two panels: the headline within-block statistic and the pooled correlation.

    Each bar carries its wavetable-bootstrap CI, and the dashed line is the 95th
    percentile of that statistic's permutation null -- i.e. where chance actually sits.
    The ceiling band is what a perfect model could reach.
    """
    panels = [HEADLINE_STAT, "rho_pooled"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5 * 2, 6))
    for ax, stat_name in zip(axes, panels):
        sub = df[df["stat"] == stat_name].sort_values("value", ascending=False)
        pos = np.arange(len(sub))
        colors = [CONTRAST_COLOR if c else SERIES_COLOR for c in sub["is_control"]]
        err = np.stack(
            [
                (sub["value"] - sub["ci_lo_wavetable"]).clip(lower=0).fillna(0),
                (sub["ci_hi_wavetable"] - sub["value"]).clip(lower=0).fillna(0),
            ]
        )
        ax.bar(pos, sub["value"], color=colors, yerr=err, capsize=3, ecolor=AXIS_COLOR)
        # Each representation has its own null, because the null depends on the block
        # structure of that representation's distances. One shared line would be wrong.
        ax.hlines(
            sub["null_p95"], pos - 0.42, pos + 0.42, color=AXIS_COLOR,
            linestyle="--", linewidth=1.2, label="permutation null, 95th pct",
        )
        ceil = ceilings.get(stat_name, np.nan)
        if np.isfinite(ceil):
            ax.axhspan(ceil, ceil + 0.02, color="#888", alpha=0.35, label="noise ceiling")
        ax.set_xticks(pos)
        ax.set_xticklabels(sub["repr"], rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(stat_name)
        ax.set_title(f"{stat_name}\n(orange = control)", fontsize=10)
        ax.legend(loc="best", frameon=False, fontsize=8, labelcolor=AXIS_COLOR)
        _style(ax)
    fig.tight_layout()
    path = os.path.join(save_dir, "fig2_scores.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    log.info(f"Saved {path}")


def plot_curves(
    d: np.ndarray, y: np.ndarray, pairs: pd.DataFrame, name: str, save_dir: str
) -> None:
    """Human curve vs model curve per modulation type -- the calc_distances.py plots with
    the listener data overlaid.

    Both series are divided by their own block mean before averaging over wavetables.
    That removes the incommensurable units (and stops one wavetable with large distances
    from dominating the model curve) WITHOUT pinning either curve at its endpoints --
    min-max normalisation would force both series to 0 at the smallest amount and 1 at
    the largest, manufacturing agreement exactly where the reader looks first. After
    this, both curves average 1.0 within each block, so what remains visible is shape:
    the model rising linearly where listeners saturate, for instance.
    """
    def norm_within_block(v: np.ndarray) -> np.ndarray:
        out = v.astype(float).copy()
        for b in np.unique(pairs["block"].to_numpy()):
            m = pairs["block"].to_numpy() == b
            # Mean of the ABSOLUTE values, so this still works if the caller passed
            # z-scored ratings, where a block mean can be zero or negative.
            mu = np.nanmean(np.abs(v[m]))
            out[m] = v[m] / mu if mu > 0 else np.nan
        return out

    dn, yn = norm_within_block(d), norm_within_block(y)
    fig, axes = plt.subplots(1, 3, figsize=(5.0 * 3, 4.6))
    for ax, mt in zip(axes, ["amp", "freq", "reg"]):
        m = pairs["mod_type"].to_numpy() == mt
        sub = pd.DataFrame(
            {"amount": pairs["amount"].to_numpy()[m], "human": yn[m], "model": dn[m]}
        )
        g = sub.groupby("amount").mean().reset_index()
        ax.plot(g["amount"], g["human"], marker="o", color=AXIS_COLOR, label="listeners")
        ax.plot(g["amount"], g["model"], marker="s", color=SERIES_COLOR, label=name)
        if mt == "freq":
            ax.set_xscale("log", base=2)
            ax.set_xticks(g["amount"])
            ax.set_xticklabels([f"{t:g}" for t in g["amount"]])
            ax.minorticks_off()
        ax.set_xlabel({"amp": "Modulation depth", "freq": "Modulation rate (Hz)",
                       "reg": "Modulation irregularity"}[mt])
        ax.set_ylabel("dissimilarity / block mean")
        ax.set_title(mt, fontsize=10)
        ax.legend(loc="best", frameon=False, fontsize=8, labelcolor=AXIS_COLOR)
        _style(ax)
    fig.suptitle(f"Listener vs {name}, mean over 6 wavetables", fontsize=11)
    fig.tight_layout()
    path = os.path.join(save_dir, "fig3_curves.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    log.info(f"Saved {path}")


# ======================================================================================
# Part 9 -- Entry point
# ======================================================================================


def compute_ceilings(mat: np.ndarray, ctx: StatContext, rng) -> Dict[str, float]:
    """Split-half noise ceiling for every statistic, plus the optimistic bound.

    Nothing in the results table means anything without these: a tau of 0.4 is close to
    perfect if listeners only reach 0.45 against each other, and poor if they reach 0.9.
    """
    ceilings = {}
    for stat_name, (stat_fn, _) in STATS.items():
        ceilings[stat_name] = split_half_ceiling(
            stat_fn,
            mat,
            ctx,
            N_SPLIT_HALVES,
            rng,
            is_correlation=stat_name.startswith(("tau", "rho")),
        )
    ceilings["rho_pooled__sb_upper"] = spearman_brown_upper(ceilings["rho_pooled"])
    log.info("Noise ceilings (split-half, conservative):")
    for k, v in ceilings.items():
        log.info(f"  {k:<26} {v:+.3f}" if np.isfinite(v) else f"  {k:<26} nan")
    if not np.isfinite(ceilings.get(HEADLINE_STAT, np.nan)) or (
        ceilings[HEADLINE_STAT] < MIN_USABLE_CEILING
    ):
        log.warning(
            f"Noise ceiling for {HEADLINE_STAT} is below MIN_USABLE_CEILING="
            f"{MIN_USABLE_CEILING}: listeners do not agree with each other, so no model "
            f"can be credited or blamed. With HUMAN_MODE='noise' this is the expected "
            f"result and confirms the ceiling estimator works."
        )
    return ceilings


def format_summary(zero_shot: pd.DataFrame, ceilings: Dict[str, float]) -> str:
    """Console table: one row per representation, the statistics as columns."""
    wide = zero_shot.pivot(index="repr", columns="stat", values="value")
    extra = zero_shot[zero_shot["stat"] == HEADLINE_STAT].set_index("repr")
    wide["p_perm"] = extra["p_perm"]
    wide["null"] = extra["null_mean"]
    wide["agree"] = extra["agreement_on_reliable"]
    wide["n_rel"] = extra["n_reliable_comparisons"]
    wide["control"] = extra["is_control"]
    cols = [
        "tau_within_block", "null", "p_perm", "rho_cross_block", "rho_pooled",
        "rho_amp", "rho_freq", "rho_reg", "r2_isotonic", "agree", "n_rel", "control",
    ]
    wide = wide[[c for c in cols if c in wide.columns]]
    wide = wide.sort_values(HEADLINE_STAT, ascending=False)
    out = wide.to_string(float_format=lambda v: f"{v:+.3f}")
    # Signed formatting is right for correlations and wrong for a p-value, so p_perm is
    # rendered separately rather than inheriting the table's formatter.
    # p is floored at 1/(n_perm + 1) by construction, so printing "0.000" would claim
    # something the test cannot support.
    def _fmt_p(v: float) -> str:
        return f"p<{1.0 / (N_PERMUTATIONS + 1):.4f}" if v <= 1.0 / (N_PERMUTATIONS + 1) else f"p={v:.3f}"

    p_str = "  ".join(
        f"{r}: {_fmt_p(v)}" for r, v in wide["p_perm"].items() if np.isfinite(v)
    )
    return f"{out}\n\n  permutation p ({HEADLINE_STAT}):  {p_str}" 


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    rng = np.random.default_rng(ANALYSIS_SEED)
    if QUICK:
        log.warning("QUICK=1: permutation and bootstrap counts reduced ~20x")

    # --- Stimuli and pairs -----------------------------------------------------------
    pairs_full = build_pairs()
    pairs_full.to_csv(os.path.join(SAVE_DIR, "pairs.csv"), index=False)

    # --- Listener data ---------------------------------------------------------------
    if HUMAN_MODE == "csv":
        assert HUMAN_RATINGS_CSV, "Set HUMAN_RATINGS_CSV when HUMAN_MODE == 'csv'"
        long = load_human_ratings(HUMAN_RATINGS_CSV)
    else:
        long = simulate_human_ratings(pairs_full, HUMAN_MODE, HUMAN_SEED)
    long.to_csv(os.path.join(SAVE_DIR, "human_ratings.csv"), index=False)
    mat_full, participants = ratings_to_matrix(long, pairs_full)

    # --- Quality control -------------------------------------------------------------
    flags, keep_part = screen_participants(mat_full, pairs_full, participants)
    flags.to_csv(os.path.join(SAVE_DIR, "participant_screening.csv"), index=False)

    # --- The analysis subset ---------------------------------------------------------
    # Self pairs are excluded here (they were needed above, as hidden references) so that
    # 18 trivially-correct points cannot carry the result.
    sel = (
        ~pairs_full["is_self"].to_numpy()
        if DROP_SELF_PAIRS
        else np.ones(len(pairs_full), bool)
    )
    pairs = pairs_full[sel].reset_index(drop=True)
    # Scale first, subset second: see apply_scaling
    mat = apply_scaling(mat_full[keep_part], PARTICIPANT_SCALING)[:, sel]
    y = np.nanmean(mat, axis=0)
    ctx = StatContext.build(pairs)
    log.info(
        f"Analysis set: {len(pairs)} pairs, {ctx.cmp_i.size} within-block comparisons, "
        f"{mat.shape[0]} participants, scaling={PARTICIPANT_SCALING!r}"
    )

    # --- Ceiling, representations, statistics ---------------------------------------
    ceilings = compute_ceilings(mat, ctx, rng)
    reprs = build_reprs(pairs_full)
    log.info("Zero-shot statistics:")
    zero_shot, dists = run_zero_shot(reprs, pairs, y, mat, ctx, ceilings, rng)
    log.info("Probe:")
    probe = run_probe(reprs, pairs, y, ctx, ceilings, rng)
    sensitivity = run_sensitivity(reprs, pairs_full, mat_full, keep_part)

    # --- Save ------------------------------------------------------------------------
    zero_shot.to_csv(os.path.join(SAVE_DIR, "zero_shot.csv"), index=False)
    probe.to_csv(os.path.join(SAVE_DIR, "probe.csv"), index=False)
    sensitivity.to_csv(os.path.join(SAVE_DIR, "sensitivity.csv"), index=False)
    pd.DataFrame([ceilings]).to_csv(
        os.path.join(SAVE_DIR, "noise_ceiling.csv"), index=False
    )

    # --- Figures ---------------------------------------------------------------------
    plot_rank_scatter(dists, y, pairs, ceilings, SAVE_DIR)
    plot_scores(zero_shot, ceilings, SAVE_DIR)
    # Curve figure for whichever non-control representation ranks highest
    head = zero_shot[(zero_shot["stat"] == HEADLINE_STAT) & (~zero_shot["is_control"])]
    if len(head):
        best = head.sort_values("value", ascending=False).iloc[0]["repr"]
        plot_curves(dists[best], y, pairs, str(best), SAVE_DIR)

    # --- Console summary -------------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"NOISE CEILING (split-half over {mat.shape[0]} participants)")
    print("=" * 88)
    for k in [HEADLINE_STAT, "rho_pooled", "rho_pooled__sb_upper", "rho_cross_block"]:
        v = ceilings.get(k, np.nan)
        print(f"  {k:<26} {v:+.3f}" if np.isfinite(v) else f"  {k:<26}    nan")
    print("\n" + "=" * 88)
    print(f"ZERO-SHOT: no fitting, sorted by {HEADLINE_STAT}")
    print("=" * 88)
    print(format_summary(zero_shot, ceilings))
    if len(probe):
        print("\n" + "=" * 88)
        print(
            "PROBE: ridge on |delta emb| (n_dims_pca = after in-fold unsupervised PCA), "
            "grouped CV"
        )
        print("=" * 88)
        print(
            probe[
                ["repr", "scheme", "n_dims_pca", "rho", "r2", "null_mean_rho",
                 "p_perm", "is_control"]
            ].to_string(index=False, float_format=lambda v: f"{v:+.3f}")
        )
    print(
        f"\nSaved tables and figures to {SAVE_DIR}\n"
        f"Reminder: 'null' is where CHANCE sits for that statistic, not zero. A value is "
        f"only interesting if it clears its own null AND is large relative to the "
        f"ceiling."
    )
    if HUMAN_MODE != "csv":
        print(
            f"\n*** HUMAN_MODE={HUMAN_MODE!r}: the listener data is PLACEHOLDER. "
            f"{'Every statistic should sit at its null and the ceiling should be ~0.' if HUMAN_MODE == 'noise' else 'Ratings were generated from the synthesis parameters, so the parameter baseline is at ceiling by construction.'} "
            f"Point HUMAN_RATINGS_CSV at the real export and set HUMAN_MODE='csv'. ***"
        )


if __name__ == "__main__":
    main()
