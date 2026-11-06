
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
│   ├── data/             # local only; downloaded recordings are not committed
│   │   ├── raw/          # matching radar .bin recording
│   │   └── reference/    # two matching ECG/PCG .csv files
│   └── results/          # diagnostic plots and summaries
├── .gitignore
└── README.md
```

The top-level Python files are controlled simulations. Scripts under `dataset_evaluation/scripts/` perform exploratory processing of recorded radar and reference data. The `data/` directories shown above are **local input folders** and may not appear in a fresh clone because raw recordings are excluded from Git.

## 5. Python Requirements and Running the Code

### Setup

This project uses Python 3 and the following packages: **NumPy**, **SciPy**, and **Matplotlib**.

Clone the repository (GitHub access is required if it is private), then install the dependencies:

```bash
git clone https://github.com/wasanaparackrama/VitalRadio-ECSE6560.git
cd VitalRadio-ECSE6560
python -m pip install numpy scipy matplotlib
```

### Run a simulation

The simulation examples can be run independently. For instance:

```bash
python vital_sign_fft.py
```

### Download and place the real radar data

Download the **3 GHz bandwidth, Position 1, Recording 1** radar recording and its **two matching ECG/PCG reference CSV files** from the [Mendeley dataset](https://data.mendeley.com/datasets/684v4r8wfr/1).

Create these local folders if necessary:

```text
dataset_evaluation/
└── data/
    ├── raw/
    │   └── adc_3GHZ_position1_ (1).bin
    └── reference/
        ├── [Target 1 reference CSV from the same recording]
        └── [Target 2 reference CSV from the same recording]
```

The bracketed CSV labels illustrate where files belong; they are **not literal filenames**. Keep the original downloaded CSV filenames and check the input paths configured in the Python scripts.

Do not mix radar and reference files from different recordings or positions.

Raw datasets are intentionally **not stored in this Git repository**. The scripts may require their input/output paths to be adjusted to match the local file layout.

### Run the recorded-data analysis

From the repository root, execute the dataset-processing scripts in sequence:

```bash
python dataset_evaluation/scripts/load_radar_data.py
python dataset_evaluation/scripts/range_processing.py
python dataset_evaluation/scripts/target_detection.py
python dataset_evaluation/scripts/phase_extraction.py
python dataset_evaluation/scripts/vital_sign_estimation.py
python dataset_evaluation/scripts/reference_validation.py
python dataset_evaluation/scripts/reference_comparison.py
```

Later scripts depend on intermediate products generated by earlier steps. These scripts describe the exploratory workflow for the example recording; a fresh clone may require matching data files and adjustments to configured file paths.

## 6. Preliminary Python Simulation Results

For an idealized 24 GHz radar phase simulation with a target at 3 m, 20 Hz slow-time sampling, and a 60-second observation window:

| Parameter | Ground truth | Estimated |
|---|---:|---:|
| Breathing rate | 15 breaths/min (0.25 Hz) | 15 breaths/min |
| Heart rate | 72 BPM (1.20 Hz) | 72 BPM |

Example motion amplitudes are 2 mm (breathing) and 0.2 mm (heartbeat).

Separate scripts explore noise, stationary clutter, and body movement. These results show recovery of preset frequencies in ideal simulated signals, not accuracy on real human subjects or hardware.

## 7. Evaluation Using an Existing FMCW Radar Dataset

**Dataset:** [FMCW radar-based multi-person vital sign monitoring data — Mendeley Data](https://data.mendeley.com/datasets/684v4r8wfr/1) (Lei et al., 2024).

For the initial evaluation, we used the **3 GHz bandwidth, Position 1, Recording 1** files from a **60 GHz TI IWR6843ISK FMCW radar**.

Our processed radar data contain:

- **1,200 slow-time frames** over 60 seconds (20 frames/s).
- **200 fast-time ADC samples** per processed frame.
- **Eight selected virtual channels**.
- Two corresponding ECG/PCG reference recordings with a documented sampling rate of **125 Hz**.

### Implemented Processing

- Decode the raw DCA1000 `.bin` radar recording and read both reference CSV files.
- Perform range FFT and examine the effects of static-clutter suppression.
- Estimate a range-angle spectrum using Minimum Variance Distortionless Response (MVDR) beamforming.
- Detect two candidate target reflections.
- Beamform candidate complex I/Q signals and extract unwrapped phase variations.
- Analyze breathing- and heartbeat-band spectral peaks using filtering and FFT.
- Detect ECG R-wave candidates in the reference recordings.
- Compare preliminary radar heartbeat-band peaks with ECG reference measurements.

### Preliminary Observations (One Recording)

| Quantity | Candidate 1 | Candidate 2 |
|---|---:|---:|
| MVDR angle | −22° | +38° |
| Estimated range | 1.25 m | 1.21 m |
| Original-phase heartbeat-band peak | 71/min | 75/min |
| Mean-cancelled heartbeat-band peak | 66/min | 102/min |

The two reference ECG recordings yield approximately:

- **Target 1:** 102 BPM
- **Target 2:** 69 BPM

With tentative angle-based subject assignments, the two-person mean absolute differences are:

- **Original-phase processing:** 14.5 BPM
- **Mean-cancelled processing:** 1.5 BPM

**Important limitations:** These observations are based on a single recording. The assignments between radar candidates and reference subjects have not been independently verified, and radar phase drift and artifacts may influence the extracted frequency peaks.

Therefore, these comparisons represent **exploratory differences, not validated heart-rate accuracy**.

Breathing-rate accuracy against independent reference data has not yet been established. The dataset's 60 GHz radar hardware also differs from the 24 GHz simulation and the planned EV-TINYRAD24G experiments.

The raw radar recordings and reference CSV files must be downloaded separately from Mendeley and kept locally.

## 8. Planned Hardware Experiments and Future Work

The remaining work will focus on further validation using real recorded radar data and planned hardware experiments.

- Validate target identities and improve radar phase extraction.
- Evaluate additional dataset recordings, target positions, and distances.
- Establish trustworthy reference breathing rates.
- Calculate validated MAE and RMSE after confirming subject assignments and processing methods.
- Improve robustness to static clutter, measurement noise, and body movement.
- Configure and characterize the **EV-TINYRAD24G** FMCW radar.
- Collect new radar measurements with independent breathing-rate and heart-rate references.
- Investigate accuracy versus distance using the available hardware.
- Compare the final results with the methodology and reported performance of the original Vital-Radio study.

Hardware differences, radar bandwidth, experimental conditions, and measurement limitations will be considered when interpreting the results.

## 9. References

1. F. Adib, H. Mao, Z. E. Kabelac, D. Katabi, and R. C. Miller, "Smart Homes that Monitor Breathing and Heart Rate," *Proceedings of CHI '15*, pp. 837–846, 2015.

2. G. Lei et al., "FMCW radar-based multi-person vital sign monitoring data," *Mendeley Data*, Version 1, 2024. https://doi.org/10.17632/684v4r8wfr.1
