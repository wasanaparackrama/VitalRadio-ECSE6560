
"""Reference ECG heartbeat analysis for the Vital-Radio project.

Dataset: Position 1, 3 GHz, Recording 1.

IMPORTANT:
The reference sampling frequency of 125 Hz is assumed from
7500 samples over 60 seconds and needs independent verification.
This script estimates ECG heartbeat rates only, not breathing rates.
Radar candidates are not yet assigned to the reference subjects.
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks


# --------------------------------------------------------
# 1. File locations and parameters
# --------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_DIR / "data" / "reference"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FS = 125.0  # Assumed Hz; verify with dataset metadata
ECG_BAND = (0.8, 25.0)

# Target 1 R-waves are predominantly negative;
# Target 2 R-waves are predominantly positive.
POLARITY = {1: -1, 2: 1}


# --------------------------------------------------------
# 2. Load reference ECG data
# --------------------------------------------------------

def load_reference(target_number):
    filename = (
        f"log_Target{target_number}_3GHZ_position1_ (1).csv"
    )
    filepath = REFERENCE_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Cannot find reference file: {filepath}"
        )

    # CSV has two header rows:
    # Column1,Column2
    # ECG,PCG
    data = np.loadtxt(
        filepath,
        delimiter=",",
        skiprows=2
    )

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Expected ECG and PCG columns")

    ecg = data[:, 0]
    pcg = data[:, 1]

    return ecg, pcg


# --------------------------------------------------------
# 3. Filter ECG and detect R-wave candidates
# --------------------------------------------------------

def estimate_heart_rate(ecg, polarity):

    sos = butter(
        3,
        ECG_BAND,
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    filtered_ecg = sosfiltfilt(sos, ecg)

    # Correct polarity for peak detection
    oriented = polarity * filtered_ecg

    peaks, _ = find_peaks(
        oriented,
        distance=int(0.45 * FS),
        prominence=1.5 * np.std(oriented)
    )

    duration = len(ecg) / FS

    # Heart rate from beat count
    heart_rate = len(peaks) * 60 / duration

    # Heart rate from median R-R interval
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / FS
        median_rr_bpm = 60 / np.median(rr_intervals)
    else:
        median_rr_bpm = float("nan")

    return filtered_ecg, peaks, heart_rate, median_rr_bpm


# --------------------------------------------------------
# 4. Main processing
# --------------------------------------------------------

def main():

    print("=" * 58)
    print("VITAL-RADIO: REFERENCE ECG HEARTBEAT ANALYSIS")
    print("=" * 58)
    print("Assumed reference sampling frequency: 125 Hz")
    print("Reference sampling rate NOT independently verified.")
    print()

    fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 7),
        sharex=True
    )

    results = []

    for target in [1, 2]:

        ecg, pcg = load_reference(target)

        filtered, peaks, bpm, median_bpm = estimate_heart_rate(
            ecg,
            POLARITY[target]
        )

        duration = len(ecg) / FS

        print(f"Target {target}")
        print("-" * 35)
        print(f"ECG samples       : {len(ecg)}")
        print(f"PCG samples       : {len(pcg)}")
        print(f"Assumed duration  : {duration:.1f} s")
        print(f"Detected ECG peaks: {len(peaks)}")
        print(f"Beat-count rate   : {bpm:.2f} BPM")
        print(f"Median R-R rate   : {median_bpm:.2f} BPM")
        print()

        results.append([
            target,
            len(ecg),
            duration,
            len(peaks),
            bpm,
            median_bpm
        ])

        # Plot first 10 seconds
        n_plot = min(int(10 * FS), len(ecg))
        time = np.arange(n_plot) / FS

        ax = axes[target - 1]

        ax.plot(
            time,
            filtered[:n_plot],
            label="Filtered ECG"
        )

        visible_peaks = peaks[peaks < n_plot]

        ax.plot(
            visible_peaks / FS,
            filtered[visible_peaks],
            "rx",
            label="Detected heartbeat peaks"
        )

        ax.set_title(
            f"Target {target}: ECG Heartbeat Detection "
            f"({bpm:.1f} BPM, preliminary)"
        )

        ax.set_ylabel("ECG amplitude (a.u.)")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time (seconds)")

    fig.suptitle(
        "Reference ECG Peak Detection - Position 1"
    )

    fig.tight_layout()

    plot_path = (
        RESULTS_DIR / "reference_ecg_heartbeat_detection.png"
    )

    fig.savefig(plot_path, dpi=160)
    plt.show()

    # Save numeric summary
    csv_path = RESULTS_DIR / "reference_ecg_summary.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Target",
            "ECG Samples",
            "Assumed Duration (s)",
            "Detected ECG Peaks",
            "Beat Count BPM",
            "Median RR BPM"
        ])

        writer.writerows(results)

    print(f"Saved plot: {plot_path}")
    print(f"Saved summary: {csv_path}")
    print()
    print("CAUTION: These are preliminary reference rates.")
    print("Verify sampling rate and subject assignment")
    print("before calculating radar estimation accuracy.")


if __name__ == "__main__":
    main()
