"""Exploratory vital-sign frequency analysis of candidate FMCW radar phase.

Dataset: Position 1, 3 GHz, Recording 1, Mendeley multi-person FMCW dataset.

Prerequisite:
    python dataset_evaluation/scripts/phase_extraction.py

Run from repository root:
    python dataset_evaluation/scripts/vital_sign_estimation.py

Inputs (local, excluded from Git):
    dataset_evaluation/data/processed/candidate_phase_signals.npz

Outputs:
    dataset_evaluation/results/vital_sign_spectra.png
    dataset_evaluation/results/vital_sign_filtered_waveforms.png
    dataset_evaluation/results/vital_sign_diagnostic_estimates.csv

Important: Reported numbers are *unverified spectral peak candidates*, NOT
validated breathing/heart rates. Candidate-to-Target1/Target2 matching, reference
sampling rate, radar phase fidelity, and subject motion must be evaluated before
calculating accuracy.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, periodogram

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PHASE_FILE = DATA_DIR / "candidate_phase_signals.npz"

# From the dataset authors' RR_BPF20.m and HR_BPF20.m design bands.
BANDS_HZ = {
    "Breathing": (0.10, 0.60),
    "Heartbeat": (0.90, 2.00),
}
FILTER_ORDER = 4  # initial exploratory filter, not identical to the authors' filters


def analyze_phase(phase: np.ndarray, fs: float, kind: str) -> dict:
    """Band-pass the phase, find largest FFT/periodogram peak within the band."""
    lo, hi = BANDS_HZ[kind]
    if fs <= 2 * hi:
        raise ValueError("Slow-time sampling must exceed twice the upper cutoff")
    sos = butter(FILTER_ORDER, (lo, hi), fs=fs, btype="bandpass", output="sos")
    clean = sosfiltfilt(sos, phase)
    f, psd = periodogram(clean, fs=fs, window="hann", detrend="constant", scaling="spectrum")
    use = (f >= lo) & (f <= hi)
    if not np.any(use):
        raise ValueError(f"No FFT frequencies in the {kind} band")
    valid_idx = np.flatnonzero(use)
    idx = valid_idx[np.argmax(psd[use])]
    return {"filtered": clean, "frequency_hz": f, "spectrum": psd,
            "peak_hz": float(f[idx]), "peak_bpm": float(60.0 * f[idx])}


def main():
    if not PHASE_FILE.exists():
        raise FileNotFoundError(f"Run phase_extraction.py first. Missing: {PHASE_FILE}")
    saved = np.load(PHASE_FILE)
    time_s = np.asarray(saved["time_s"], dtype=float)
    if len(time_s) < 20:
        raise ValueError("Too few phase samples")
    dt = float(np.median(np.diff(time_s)))
    if not np.allclose(np.diff(time_s), dt, atol=1e-8):
        raise ValueError("This FFT approach assumes evenly spaced radar frames")
    fs = 1 / dt
    phases = {
        "Original": np.asarray(saved["phase_dynamic"], dtype=float),
        "Mean cancelled": np.asarray(saved["mean_cancelled_phase_dynamic"], dtype=float),
    }
    angles = saved["candidate_angles_deg"]
    ranges = saved["candidate_ranges_m"]
    n_targets = len(angles)
    for name, arr in phases.items():
        if arr.shape != (n_targets, len(time_s)) or not np.isfinite(arr).all():
            raise ValueError(f"Unexpected data shape/nonfinite values for {name}: {arr.shape}")
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    all_results = {}
    records = []
    print("=== Exploratory radar phase frequency analysis (NOT validated vital signs) ===")
    print(f"{len(time_s)} samples, {fs:.2f} Hz sampling, {len(time_s)/fs:.1f} s duration")
    print(f"Nominal FFT bin spacing = {60.0 * fs / len(time_s):.2f} BPM")
    for i in range(n_targets):
        label = f"Candidate {i+1} ({angles[i]:+.0f}°, {ranges[i]:.2f} m)"
        print(f"\n{label}")
        for method, phase_set in phases.items():
            for kind in BANDS_HZ:
                v = analyze_phase(phase_set[i], fs, kind)
                all_results[i, method, kind] = v
                print(f"  {method:16s} {kind:9s}: peak at {v['peak_hz']:.3f} Hz "
                      f"({v['peak_bpm']:.0f} per minute)")
                records.append({
                    "candidate": i+1,
                    "estimated_angle_deg": f"{angles[i]:.1f}",
                    "estimated_range_m": f"{ranges[i]:.3f}",
                    "preprocessing": method,
                    "frequency_band": kind,
                    "spectral_peak_hz": f"{v['peak_hz']:.5f}",
                    "spectral_peak_per_min": f"{v['peak_bpm']:.2f}",
                    "status": "UNVALIDATED spectral candidate (not ground-truth checked)",
                })

        for kind in BANDS_HZ:
            a = all_results[i, "Original", kind]["peak_bpm"]
            b = all_results[i, "Mean cancelled", kind]["peak_bpm"]
            if abs(a-b) > 0.01:
                print(f"  WARNING: {kind} peak differs by {abs(a-b):.0f} per min "
                      "between preprocessing choices")

    output_csv = RESULTS_DIR / "vital_sign_diagnostic_estimates.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    # FFT diagnostic: 2 candidates x 2 frequency bands, overlay both methods.
    fig, axes = plt.subplots(n_targets, 2, figsize=(12, 7), constrained_layout=True)
    for i in range(n_targets):
        for j, kind in enumerate(BANDS_HZ):
            ax = axes[i, j]
            for method in phases:
                v = all_results[i, method, kind]
                f = v["frequency_hz"]
                mask = (f >= 0) & (f <= BANDS_HZ[kind][1] + 0.15)
                relative_db = 10*np.log10(np.maximum(v["spectrum"] / max(np.max(v["spectrum"]), 1e-30), 1e-12))
                ax.plot(f[mask]*60, relative_db[mask], label=f"{method} ({v['peak_bpm']:.0f}/min)", lw=1.4)
            ax.set_xlim(BANDS_HZ[kind][0]*60, BANDS_HZ[kind][1]*60)
            ax.set_ylim(-75, 2)
            ax.set_title(f"Candidate {i+1}: {kind} band")
            ax.set_xlabel("Frequency (cycles/min)")
            ax.set_ylabel("Relative spectral power (dB)")
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle("Exploratory radar phase spectra — spectral peaks NOT validated as vital signs")
    path_fft = RESULTS_DIR / "vital_sign_spectra.png"
    fig.savefig(path_fft, dpi=160)

    # Compare bandpassed waveforms (original signal only). Pre/post not comparable in amplitude.
    fig, axes = plt.subplots(n_targets, 2, figsize=(12, 7), constrained_layout=True, sharex=True)
    for i in range(n_targets):
        for j, kind in enumerate(BANDS_HZ):
            ax = axes[i,j]
            v = all_results[i, "Original", kind]
            ax.plot(time_s, v["filtered"], lw=.9)
            ax.set_title(f"Candidate {i+1}: {kind} band-pass phase")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Filtered phase (rad)")
            ax.grid(alpha=.25)
    fig.suptitle("Band-passed radar phase — periodicity exploratory, subject identity not verified")
    path_time = RESULTS_DIR / "vital_sign_filtered_waveforms.png"
    fig.savefig(path_time, dpi=160)

    print(f"\nSaved: {output_csv}\nSaved: {path_fft}\nSaved: {path_time}")
    print("NOT YET VALIDATED: reference sampling/synchronization, person assignment, and artifact rejection.")
    print("Do not claim accurate breathing/heartbeat estimation from these peaks yet.")
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
