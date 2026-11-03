"""Exploratory FMCW range analysis for Mendeley Position 1, 3 GHz, test 1.

Run from the repository root:
    python dataset_evaluation/scripts/range_processing.py

Requires load_radar_data.py in the same scripts directory, and the raw BIN
in dataset_evaluation/data/raw. This script does not detect both persons
or estimate breathing/heart rates.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from load_radar_data import (
    ADC_RATE_HZ,
    CHIRP_SLOPE_HZ_PER_S,
    FRAME_PERIOD_S,
    RADAR_FILE,
    SPEED_OF_LIGHT,
    load_dca1000,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"
NFFT = 256
RANGE_LIMIT_M = 5.0
EXPECTED_DISTANCE_M = 1.0  # Dataset documentation, Position 1: both subjects at 1 m


def main():
    if not RADAR_FILE.is_file():
        raise FileNotFoundError(f"Radar recording not found: {RADAR_FILE}")

    radar = load_dca1000(RADAR_FILE)  # [1200 frames, 200 ADC samples, 8 channels]
    frames, samples, channels = radar.shape
    print(f"Radar data loaded: {frames} frames x {samples} samples x {channels} channels")

    # Fast-time range FFT: one FFT for each frame/channel.
    # A Hann window reduces leakage from strong neighboring reflections.
    win = np.hanning(samples)[None, :, None]
    spectra = np.fft.fft(radar * win, n=NFFT, axis=1)[:, :NFFT // 2, :]

    # As in the authors' MTI.m, subtract stationary components across frames.
    # This is used ONLY for exploratory motion-sensitive range plots.
    spectra_dynamic = spectra - np.mean(spectra, axis=0, keepdims=True)

    freq_hz = np.arange(NFFT // 2) * ADC_RATE_HZ / NFFT
    range_m = SPEED_OF_LIGHT * freq_hz / (2 * CHIRP_SLOPE_HZ_PER_S)
    display_mask = range_m <= RANGE_LIMIT_M

    # Average magnitudes across all frames and selected virtual channels.
    magnitude_all = np.mean(np.abs(spectra), axis=(0, 2))
    magnitude_dynamic = np.mean(np.abs(spectra_dynamic), axis=(0, 2))
    reference = max(np.max(magnitude_all), 1e-12)
    original_db = 20 * np.log10(np.maximum(magnitude_all, 1e-12) / reference)
    dynamic_db = 20 * np.log10(np.maximum(magnitude_dynamic, 1e-12) / reference)

    # The two subjects are both near 1 m in Position 1. This is a candidate
    # motion-sensitive range peak, NOT verified subject detection.
    candidate_mask = (range_m >= 0.4) & (range_m <= 2.0)
    candidate_bins = np.flatnonzero(candidate_mask)
    candidate_idx = candidate_bins[np.argmax(magnitude_dynamic[candidate_mask])]
    candidate_m = range_m[candidate_idx]
    print(f"Range FFT size: {NFFT}")
    print(f"Nominal range resolution c/(2B): {SPEED_OF_LIGHT / (2 * 3e9):.3f} m")
    print(f"Range FFT grid spacing: {range_m[1] - range_m[0]:.3f} m")
    print(f"Expected subject distance (Position 1): {EXPECTED_DISTANCE_M:.2f} m each")
    print(f"Strongest motion-sensitive peak in 0.4–2.0 m: {candidate_m:.3f} m")
    print("WARNING: This peak is not yet a confirmed person; two subjects require angle separation.")

    # A range-vs-time view of moving reflections after slow-time mean removal.
    dynamic_range_time = np.mean(np.abs(spectra_dynamic), axis=2)
    heat = 20 * np.log10(np.maximum(dynamic_range_time[:, display_mask], 1e-12))
    # Relative scale makes plots easier to compare; ignore absolute units.
    heat -= np.max(heat)
    t_end = frames * FRAME_PERIOD_S

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    axes[0].plot(range_m[display_mask], original_db[display_mask], label="Before slow-time mean cancellation", lw=1.5)
    axes[0].plot(range_m[display_mask], dynamic_db[display_mask], label="After slow-time mean cancellation", lw=1.5)
    axes[0].axvline(EXPECTED_DISTANCE_M, color="black", linestyle="--", lw=1, label="Nominal subject distance (1 m)")
    axes[0].axvline(candidate_m, color="gray", linestyle=":", lw=1.2, label=f"Candidate moving peak ({candidate_m:.2f} m)")
    axes[0].set_title("FMCW range profiles averaged across 8 selected virtual channels")
    axes[0].set_xlabel("Range (m)")
    axes[0].set_ylabel("Relative FFT magnitude (dB)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    vmax = np.percentile(heat, 99)
    vmin = np.percentile(heat, 5)
    im = axes[1].imshow(
        heat,
        extent=(range_m[display_mask][0], range_m[display_mask][-1], t_end, 0),
        interpolation="nearest", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
    )
    axes[1].axvline(EXPECTED_DISTANCE_M, color="white", linestyle="--", lw=1)
    axes[1].set_title("Moving-reflector energy over time (mean cancellation)")
    axes[1].set_xlabel("Range (m)")
    axes[1].set_ylabel("Time (s)")
    fig.colorbar(im, ax=axes[1], label="Relative magnitude (dB)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = OUTPUT_DIR / "range_processing.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")
    plt.show()


if __name__ == "__main__":
    main()
