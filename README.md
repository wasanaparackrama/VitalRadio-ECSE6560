# VitalRadio: Contactless Breathing and Heart Rate Estimation Using FMCW Radar

**ECSE 6560 – Modern Communication Systems**
Rensselaer Polytechnic Institute (RPI)
Fall 2026

## 1. Project Overview

This project investigates contactless human breathing-rate and heart-rate estimation using Frequency-Modulated Continuous-Wave (FMCW) radar.

The project is based on the research paper **"Smart Homes that Monitor Breathing and Heart Rate" (Vital-Radio)** by Adib et al. (2015).

The main objective is to independently implement and evaluate the fundamental FMCW radar signal-processing methods used to detect small human chest movements caused by breathing and heartbeat.

The project uses two evaluation approaches:

1. **Python-based simulation:** Develop and validate FMCW radar signal-processing algorithms using simulated radar signals with known ground-truth values.
2. **Existing radar dataset:** Evaluate breathing-rate and heart-rate estimation using publicly available FMCW radar measurements and reference vital-sign data.

## 2. Project Objectives

- Model FMCW transmitted and received radar signals.
- Generate beat signals and estimate target distance using the Fast Fourier Transform (FFT).
- Simulate human chest movements caused by breathing and heartbeat.
- Extract phase variations from complex radar I/Q signals.
- Estimate breathing rate and heart rate using frequency-domain analysis.
- Investigate the effects of noise, static clutter, and body movement.
- Evaluate estimation performance using existing FMCW radar datasets.

## 3. Signal-Processing Methodology

The basic signal-processing pipeline is:

```text
FMCW Chirp Transmission
          |
          v
Received Radar Signal
          |
          v
Beat Signal Generation
          |
          v
Range FFT and Target Detection
          |
          v
Complex I/Q Signal
          |
          v
Phase Extraction and Unwrapping
          |
          v
Vital-Sign Filtering and FFT
          |
          v
Breathing Rate and Heart Rate
```

### FMCW Range Estimation

The FMCW beat frequency is related to the target range by:

[
f_b = \frac{2SR}{c}
]

Therefore,

[
R = \frac{cf_b}{2S}
]

where:

- (R): Target distance
- (S): FMCW chirp slope
- (c): Speed of light
- (f_b): Beat frequency

### Human Chest Motion

Human chest displacement is modeled as:

[
x(t)=A_B\cos(2\pi f_Bt)+A_H\cos(2\pi f_Ht)
]

where the two components represent breathing-related and heartbeat-related movements.

### Radar Phase

Small chest movements produce variations in the phase of the reflected radar signal:

[
\phi(t)=\frac{4\pi R(t)}{\lambda}
]

The phase variations are analyzed to estimate breathing and heartbeat frequencies.

The estimated rates are:

[
BR=60f_B
]

[
HR=60f_H
]

where (BR) is breathing rate in breaths per minute and (HR) is heart rate in beats per minute.

## 4. Repository Structure

```text
VitalRadio-ECSE6560/
|
|-- fmcw_transmitter.py
|-- received_signal.py
|-- beat_signal.py
|-- range_fft.py
|
|-- chest_motion.py
|-- phase_from_chest_motion.py
|-- vital_sign_fft.py
|
|-- noise_analysis.py
|-- static_clutter.py
|-- clutter_removal.py
|-- vital_sign_filtering.py
|-- body_motion.py
|
|-- README.md
```

### Python File Descriptions

| File                         | Description                                                                       |
| ---------------------------- | --------------------------------------------------------------------------------- |
| `fmcw_transmitter.py`        | Generates FMCW chirp signals and visualizes instantaneous frequency.              |
| `received_signal.py`         | Simulates reflected radar signals with propagation delay.                         |
| `beat_signal.py`             | Generates the beat signal by mixing transmitted and received signals.             |
| `range_fft.py`               | Applies FFT to the beat signal to estimate target distance.                       |
| `chest_motion.py`            | Simulates periodic breathing and heartbeat displacement.                          |
| `phase_from_chest_motion.py` | Converts chest displacement into radar phase variations and complex I/Q signals.  |
| `vital_sign_fft.py`          | Estimates breathing and heartbeat frequencies using FFT.                          |
| `noise_analysis.py`          | Investigates estimation performance under different signal-to-noise ratios.       |
| `static_clutter.py`          | Simulates the effect of stationary reflections on the received radar signal.      |
| `clutter_removal.py`         | Investigates removal of static reflections using complex-signal mean subtraction. |
| `vital_sign_filtering.py`    | Applies bandpass filters to separate breathing and heartbeat components.          |
| `body_motion.py`             | Investigates interference caused by larger body movements.                        |

**Note:** The current scripts implement and test individual signal-processing components. A fully integrated multi-chirp FMCW processing pipeline is planned for further development.

## 5. Simulation Parameters

The following parameters are used in the simulated vital-sign estimation examples.

| Parameter                        | Value          |
| -------------------------------- | -------------- |
| Radar carrier frequency          | 24 GHz         |
| Target distance                  | 3 m            |
| Breathing frequency              | 0.25 Hz        |
| Breathing rate                   | 15 breaths/min |
| Heartbeat frequency              | 1.20 Hz        |
| Heart rate                       | 72 BPM         |
| Breathing displacement amplitude | 2 mm           |
| Heartbeat displacement amplitude | 0.2 mm         |
| Slow-time sampling frequency     | 20 Hz          |
| Observation duration             | 60 seconds     |

The FMCW transmitter and range-processing examples use illustrative simulation parameters that are not intended to represent the final hardware configuration.

## 6. Preliminary Simulation Results

Under the ideal simulated conditions, the frequency-domain method successfully recovers the known breathing and heartbeat frequencies.

| Measurement    | Ground Truth   | Estimated      |
| -------------- | -------------- | -------------- |
| Breathing rate | 15 breaths/min | 15 breaths/min |
| Heart rate     | 72 BPM         | 72 BPM         |

The initial simulations also examine how measurement noise, static reflections, and body movement affect the extracted radar phase and estimated vital signs.

These are preliminary simulation results rather than performance measurements from real radar recordings.

## 7. Python Requirements

The implementation uses:

- Python 3
- NumPy
- SciPy
- Matplotlib

Install the required packages using:

```bash
python -m pip install numpy scipy matplotlib
```

### Running the Simulations

Clone the repository:

```bash
git clone https://github.com/wasanaparackrama/VitalRadio-ECSE6560.git
```

Open the project directory:

```bash
cd VitalRadio-ECSE6560
```

Run an individual simulation:

```bash
python fmcw_transmitter.py
```

For example, to estimate breathing rate and heart rate:

```bash
python vital_sign_fft.py
```

To investigate noise:

```bash
python noise_analysis.py
```

Each script can currently be executed independently.

## 8. Evaluation Using an Existing Dataset

The next stage will use a publicly available FMCW radar dataset containing real radar measurements and reference vital-sign signals.

**Dataset:** FMCW radar-based multi-person vital sign monitoring data

**Source:** Mendeley Data

**Link:**&#x20;

[https://data.mendeley.com/datasets/684v4r8wfr/1](https://data.mendeley.com/datasets/684v4r8wfr/1)

The planned evaluation includes:

1. Reading recorded FMCW radar data.
2. Processing radar signals to identify human targets.
3. Extracting phase variations associated with breathing and heartbeat.
4. Estimating breathing and heart rates.
5. Comparing estimated rates with the reference measurements.
6. Evaluating estimation error and accuracy under different recording conditions.

This dataset-based processing is planned and is not yet included in the current repository.

## 9. Evaluation Metrics

The main evaluation metrics are:

- **Mean Absolute Error (MAE):** Average absolute difference between estimated and reference vital-sign rates.
- **Root Mean Square Error (RMSE):** Measures estimation error while placing greater weight on larger deviations.
- **Relative Error:** Estimation error relative to the reference measurement.
- **Estimation Accuracy:** Percentage accuracy based on relative estimation error.

The metrics will be evaluated separately for breathing rate and heart rate.

## 10. Future Work

- Integrate FMCW range processing and slow-time vital-sign processing into a complete simulation pipeline.
- Improve robustness to noise, static clutter, and body movement.
- Implement Python processing for the Mendeley radar dataset.
- Compare estimated vital signs with reference measurements.
- Evaluate performance at different distances and under different recording conditions.
- Compare the resulting performance with the methodology and reported results of the original Vital-Radio study.

## 11. References

[1] F. Adib, H. Mao, Z. E. Kabelac, D. Katabi, and R. C. Miller, "Smart Homes that Monitor Breathing and Heart Rate," *Proceedings of the 33rd Annual ACM Conference on Human Factors in Computing Systems (CHI '15)*, pp. 837–846, 2015.

[2] Mendeley Data, "FMCW radar-based multi-person vital sign monitoring data."&#x20;

[https://data.mendeley.com/datasets/684v4r8wfr/1](https://data.mendeley.com/datasets/684v4r8wfr/1)

---

**Project:** ECSE 6560 – Modern Communication Systems
**Institution:** Rensselaer Polytechnic Institute
**Academic Term:** Fall 2026
