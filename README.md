# Human Perception of Modulated Synthesizers

An investigation into how listeners perceive changes in wavetable synthesizer sounds as the **amount** and **type** of modulation vary.

This repository contains the audio stimuli, acoustic-feature analysis, anonymised WebMUSHRA listening-test data, and statistical analysis developed for my MSc project in **Sound and Music Computing** at **Queen Mary University of London**.

## Research question

How sensitive are listeners to different amounts of modulation applied to wavetable position, and does that sensitivity depend on the modulation type, timbral feature, or source sound?

> **Terminology note:** amplitude, frequency, and irregularity describe properties of the control signal used to move through a wavetable. They should not be interpreted as conventional audio-rate amplitude modulation (AM) or frequency modulation (FM).

## Study design

The experiment compared three modulation dimensions across three timbral groupings and two source types:

| Factor | Levels |
| --- | --- |
| Modulation | Amplitude, frequency, irregularity |
| Timbral grouping | Warmth, brightness, richness |
| Source | Real/natural, synthetic |
| Modulation amount | Five values per modulation dimension |

Six representative wavetables were used: one real/natural and one synthetic source for each timbral grouping. Each stimulus was a four-second monophonic wavetable sweep, loudness-normalised to **−18 LUFS**.

The listening test used a WebMUSHRA-style interface with 18 conditions (3 modulation types × 3 timbral groupings × 2 source types). In each condition, participants rated five modulated versions against a minimally modulated reference on a scale from 0 (*identical*) to 100 (*completely different*).

### Modulation values

| Level | Amplitude | Frequency (Hz) | Irregularity (%) |
| ---: | ---: | ---: | ---: |
| 0 | 0.1 | 0.25 | 0.0 |
| 1 | 0.3 | 0.50 | 12.5 |
| 2 | 0.5 | 1.00 | 25.0 |
| 3 | 0.7 | 2.00 | 37.5 |
| 4 | 0.9 | 4.00 | 50.0 |

## Main findings

- Perceived difference increased approximately linearly with modulation amount.
- Frequency modulation of the wavetable-position control signal produced the largest perceived differences.
- Amplitude modulation produced intermediate differences.
- Irregularity modulation produced the smallest differences.
- The same ordering—**frequency > amplitude > irregularity**—was observed at both low and high modulation amounts.

In the final two-way repeated-measures analysis (*N* = 52), there were significant effects of modulation type and amount group, together with a significant but comparatively small interaction:

- modulation: *F*(2, 102) = 115.758, *p* = 5.75 × 10⁻²⁷, ges = 0.430;
- amount group: *F*(1, 51) = 1044.892, *p* = 1.22 × 10⁻³⁵, ges = 0.659;
- modulation × amount group: *F*(2, 102) = 21.040, *p* = 2.24 × 10⁻⁸, ges = 0.027.

## Repository structure

```text
.
├── audio_files/                    # Rendered wavetable sweep stimuli
├── OutputEDA_Features/             # Acoustic-descriptor visualisations
│   └── wavetable_descriptors.csv   # Extracted descriptor values
├── code/
│   ├── MushraDataAnalysis.R        # Filtering, plots and statistical tests
│   ├── features.py                 # Differentiable audio-feature extractors
│   ├── mod_sig_metrics.py           # Modulation-signal metrics
│   ├── mod_signal_work.ipynb        # Modulation analysis notebook
│   └── mushra.csv                   # Anonymised listening-test responses
└── README.md
```

The feature code includes loudness, spectral centroid, spectral spread, spectral flatness, spectral flux, temporal centroid, and amplitude-envelope utilities. The modulation utilities include range, entropy, spectral entropy, total variation, and turning-point metrics.

## Reproducing the analysis

### R analysis

The main listening-test analysis is in `code/MushraDataAnalysis.R`.

Install the required R packages:

```r
install.packages(c(
  "broom",
  "dplyr",
  "emmeans",
  "ggh4x",
  "ggplot2",
  "ggsignif",
  "purrr",
  "rstatix",
  "tidyr"
))
```

Before running the script, replace its local data path:

```r
mushra <- read.csv("code/mushra.csv")
```

Then run the file from the repository root in R or RStudio. The script contains the quality-screening rules, participant-level aggregation, linear-model comparisons, repeated-measures ANOVAs, paired comparisons, and figure generation used in the project.

### Python utilities

The Python modules require Python 3 and the following core packages:

```bash
python -m pip install numpy pyloudnorm torch torchaudio
```

They can then be imported from the `code` directory, for example:

```python
from features import Loudness, SpectralCentroid, SpectralFlatness
from mod_sig_metrics import SpectralEntropyMetric, TotalVariationMetric
```

## Data screening

The analysis excludes training trials and applies quality checks for unsuitable playback devices, poor identification of the hidden reference, identical responses, very short completion times, and insufficient rating variation. Two participant records with incomplete condition coverage are removed from analyses requiring a consistent sample of 52 participants. Analyses spanning all 18 experimental conditions use only participants with complete data for those conditions.

Although the dataset uses session UUIDs rather than participant names, it should still be treated as research data and handled responsibly.

## Author

**Sachin Subramanian**  
MSc Sound and Music Computing, Queen Mary University of London

## Acknowledgements

This work was conducted at Queen Mary University's Centre for Digital Music. Thanks to the project supervisors and mentors who supported the study, and to everyone who participated in the listening experiment.

## Citation

If you use this repository, please cite it as:

```text
Subramanian, S. (2026). Human Perception of Modulated Synthesizers.
GitHub repository: https://github.com/Sachinsub0/HumanPerceptionOfModulatedSynthesizers
```

## Licence

No licence has currently been specified. Unless a licence is added, the repository's contents remain under the author's default copyright.
