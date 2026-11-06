# VitalRadio: Contactless Breathing and Heart-Rate Estimation Using FMCW Radar

**ECSE 6560 – Modern Communication Systems**  
Rensselaer Polytechnic Institute (RPI) | Fall 2026

## 1. Project Overview

This course project studies contactless breathing-rate and heart-rate estimation using **frequency-modulated continuous-wave (FMCW) radar**. It is a reproduction-oriented implementation of core signal-processing ideas from *Smart Homes that Monitor Breathing and Heart Rate* (Vital-Radio) by Adib et al. (2015).

Our work has three complementary parts:

1. **Python simulation:** Model FMCW signals, chest motion, phase changes, and vital-sign frequency estimation with known ground truth.
2. **Existing radar dataset:** Process real, publicly available FMCW radar recordings and compare exploratory estimates against physiological reference signals.
3. **Planned hardware experiments:** Evaluate the methods using the Analog Devices **EV-TINYRAD24G** radar and independent reference measurements.

**Project status:** The simulation scripts and an initial real-data processing pipeline have been implemented. The dataset-based results are **preliminary**; robust measurement accuracy and hardware experiments remain future work. This repository does not claim a full reproduction of the original paper's results.

## 2. Project Objectives

- Model FMCW chirp transmission, propagation delay, reflected signals, beat-signal generation, and range estimation.
- Model breathing- and heartbeat-related chest displacement and the resulting radar phase variation.
- Recover known breathing and heartbeat frequencies in controlled Python simulations.
- Apply range FFT, clutter suppression, and range-angle processing to recorded FMCW radar data.
- Extract complex I/Q signals and analyze unwrapped phase for vital-sign frequencies.
- Compare radar-derived frequency peaks with independent reference measurements, and investigate sources of error.
- Extend the evaluation to more recordings and the planned EV-TINYRAD24G hardware experiments.

## 3. Signal-Processing Methodology

```text
FMCW transmission and reflected signal
                 |
                 v
           Beat signal
                 |
                 v
             Range FFT
                 |
                 v
     Target/range-angle selection
                 |
                 v
      Complex I/Q phase extraction
                 |
                 v
      Phase unwrapping and filtering
                 |
                 v
      Frequency analysis (FFT)
                 |
                 v
      Breathing / heartbeat bands
                 |
                 v
      Reference-data comparison
```

The basic FMCW range relation is:

- `f_beat = (2 * S * R) / c`
- `R = (c * f_beat) / (2 * S)`

where `S` is chirp slope, `R` is range, and `c` is the speed of light.

In the simplified chest-motion model:

- `x(t) = A_B * cos(2π * f_B * t) + A_H * cos(2π * f_H * t)`
- `R(t) = R0 + x(t)`
- `φ(t) = (4 * π * R(t)) / λ`
- `z(t) = A * exp(j * φ(t))`

Here, `A_B`, `f_B` are breathing-motion amplitude and frequency; `A_H`, `f_H` are heartbeat-motion amplitude and frequency; `λ` is the radar wavelength. Estimated rates are calculated as `BR = 60 * f_B` (breaths/min) and `HR = 60 * f_H` (beats/min).

The early simulation scripts test these components individually; they are not yet one fully integrated multi-chirp FMCW simulator.

## 4. Repository Structure

```text
VitalRadio-ECSE6560/
├── fmcw_transmitter.py
├── received_signal.py
├── beat_signal.py
├── range_fft.py
├── chest_motion.py
├── phase_from_chest_motion.py
├── vital_sign_fft.py
├── noise_analysis.py
├── static_clutter.py
├── clutter_removal.py
├── vital_sign_filtering.py
├── body_motion.py
├── dataset_evaluation/
│   ├── scripts/
│   │   ├── load_radar_data.py
│   │   ├── range_processing.py
│   │   ├── target_detection.py
│   │   ├── phase_extraction.py
│   │   ├── vital_sign_estimation.py
│   │   ├── reference_validation.py
│   │   └── reference_comparison.py
│   ├── data/             # downloaded/derived data; keep local
│   └── results/          # diagnostic plots and summaries
├── .gitignore
└── README.md
```

The top-level Python files are the controlled simulations. Scripts under `dataset_evaluation/scripts/` perform the exploratory processing of recorded radar and reference data.

## 5. Python Requirements and Running the Code

Required Python packages: **NumPy**, **SciPy**, and **Matplotlib**.

```bash
python -m pip install numpy scipy matplotlib
```

For example, run the ideal vital-sign simulation:

```bash
python vital_sign_fft.py
```

After downloading the matching dataset files to the local `dataset_evaluation/data/` folders, run the real-data processing scripts from the repository root:

```bash
python dataset_evaluation/scripts/load_radar_data.py
python dataset_evaluation/scripts/range_processing.py
python dataset_evaluation/scripts/target_detection.py
python dataset_evaluation/scripts/phase_extraction.py
python dataset_evaluation/scripts/vital_sign_estimation.py
python dataset_evaluation/scripts/reference_validation.py
python dataset_evaluation/scripts/reference_comparison.py
```

The later scripts depend on data products generated by earlier steps. The scripts are currently configured for the example recording described below.

## 6. Preliminary Python Simulation Results

For an idealized 24 GHz radar phase simulation with a target at 3 m, 20 Hz slow-time sampling, and a 60-second observation window:

| Parameter | Ground truth | Estimated |
|---|---:|---:|
| Breathing rate | 15 breaths/min (0.25 Hz) | 15 breaths/min |
| Heart rate | 72 BPM (1.20 Hz) | 72 BPM |

Example motion amplitudes are 2 mm (breathing) and 0.2 mm (heartbeat). Separate scripts explore noise, stationary clutter, and body movement. These results show recovery of preset frequencies in ideal simulated signals, not accuracy on real patients or hardware.

## 7. Evaluation Using an Existing FMCW Radar Dataset

**Dataset:** [FMCW radar-based multi-person vital sign monitoring data — Mendeley Data](https://data.mendeley.com/datasets/684v4r8wfr/1) (Lei et al., 2024).

For the initial evaluation, we used the **3 GHz bandwidth, Position 1, Recording 1** files from a **60 GHz TI IWR6843ISK FMCW radar**. The radar data contain **1,200 frames over 60 seconds** (20 frames/s), **200 ADC samples per frame**, and **eight selected virtual channels** in our processing. Two accompanying ECG/PCG CSV files provide physiological reference waveforms; the documented reference sample rate is 125 Hz.

### Implemented processing

- Decode the raw DCA1000 `.bin` radar recording and read both reference CSVs.
- Perform range FFT and examine the effect of static-clutter suppression.
- Estimate a range-angle spectrum with MVDR beamforming and detect two candidate reflections.
- Beamform candidate I/Q signals, unwrap phase, and analyze breathing- and heartbeat-band spectral peaks.
- Detect ECG R-wave candidates in the reference signals and compare preliminary heartbeat-band peaks under consistent preprocessing choices.

### Preliminary observations (one recording)

| Quantity | Candidate 1 | Candidate 2 |
|---|---:|---:|
| MVDR angle | −22° | +38° |
| Estimated range | 1.25 m | 1.21 m |
| Original-phase heartbeat-band peak | 71/min | 75/min |
| Mean-cancelled heartbeat-band peak | 66/min | 102/min |

The two reference ECG recordings yield approximately **102 BPM** (Target 1) and **69 BPM** (Target 2). With **tentative angle-based assignments**, the two-person mean absolute differences are **14.5/min** for original-phase processing and **1.5/min** for mean-cancelled processing.

**Important limitations:** These numbers are from a single recording. The assignments between radar candidates and reference subjects have not been independently verified, and phase drift/artifacts affect the spectral peaks. Therefore, the comparisons are **exploratory differences, not validated heart-rate accuracy**. Breathing-rate accuracy against reference data has not yet been established. The dataset's 60 GHz hardware also differs from both the 24 GHz simulation and the planned EV-TINYRAD24G experiments.

The raw recordings and reference CSV files should be downloaded from Mendeley and kept locally; do not commit bulky dataset files into this repository.

## 8. Planned Hardware Experiments and Future Work

- Validate target identities and phase extraction; test additional positions, distances, and recordings.
- Establish trustworthy breathing reference rates and calculate validated MAE/RMSE once assignments and methods are checked.
- Improve robustness to static clutter, noise, and incidental body movement.
- Integrate and characterize the **EV-TINYRAD24G** radar for new measurements with independent vital-sign references.
- Compare the final evaluation methodology and results with the original Vital-Radio study, while accounting for differences in radar hardware, bandwidth, and experimental conditions.

## 9. References

1. F. Adib, H. Mao, Z. E. Kabelac, D. Katabi, and R. C. Miller, "Smart Homes that Monitor Breathing and Heart Rate," *Proceedings of CHI '15*, pp. 837–846, 2015.
2. G. Lei et al., "FMCW radar-based multi-person vital sign monitoring data," *Mendeley Data*, Version 1, 2024. https://data.mendeley.com/datasets/684v4r8wfr/1
