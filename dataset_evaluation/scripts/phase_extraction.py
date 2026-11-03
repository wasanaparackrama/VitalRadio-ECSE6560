"""Explore radar phase for the two candidate reflectors in Position 1 / 3 GHz.

Run from the repository root:
    python dataset_evaluation/scripts/phase_extraction.py

Requires:
    load_radar_data.py and target_detection.py in the same scripts folder
    numpy, scipy, matplotlib

Inputs (kept local, not committed):
    dataset_evaluation/data/raw/adc_3GHZ_position1_ (1).bin

Outputs:
    dataset_evaluation/results/phase_extraction.png
    dataset_evaluation/results/phase_preprocessing_comparison.png
    dataset_evaluation/data/processed/candidate_phase_signals.npz

Method: detect two range-angle candidates from slow-time-mean-cancelled data,
compute MVDR weights at each selected range bin, then apply the weights to
both the ORIGINAL complex range FFT and its mean-cancelled version.

IMPORTANT: These are exploratory candidate phase signals, not yet proven
breathing or heartbeat signals. Target1/Target2 identity, array calibration,
clutter leakage, phase fidelity, and reference synchronization remain unverified.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend

from load_radar_data import FRAME_PERIOD_S, RADAR_FILE, load_dca1000
from target_detection import (
    DIAGONAL_LOADING,
    NFFT,
    compute_range_angle_mvdr,
    find_two_candidates,
)

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
PROCESSED_DIR = BASE_DIR / "data" / "processed"  # ignored by .gitignore


def make_range_fft(radar: np.ndarray) -> np.ndarray:
    """Complex range FFT of original chirps: [frames, range_bins, channels]."""
    if radar.ndim != 3 or radar.shape[2] != 8:
        raise ValueError(f"Expected [frames, adc_samples, 8], got {radar.shape}")
    window = np.hanning(radar.shape[1])[None, :, None]
    return np.fft.fft(radar * window, n=NFFT, axis=1)[:, :NFFT // 2, :]


def beamform_candidate(
    range_fft: np.ndarray, range_m: np.ndarray, candidate: dict
) -> dict:
    """Derive MVDR weights from mean-cancelled snapshots, use original I/Q for phase."""
    bin_index = int(np.argmin(np.abs(range_m - candidate["range_m"])))
    angle_deg = float(candidate["angle_deg"])
    snapshots = np.asarray(range_fft[:, bin_index, :], dtype=np.complex128)
    centered = snapshots - snapshots.mean(axis=0, keepdims=True)
    n_frames, antennas = centered.shape

    # Same covariance convention as compute_range_angle_mvdr():
    # R[i,j] = E[x_i * conj(x_j)].
    covariance = (centered.T @ centered.conj()) / n_frames
    mean_power = float(np.trace(covariance).real / antennas)
    covariance += (DIAGONAL_LOADING * mean_power + 1e-8) * np.eye(antennas)

    steering = np.exp(
        -1j * np.pi * np.arange(antennas) * np.sin(np.deg2rad(angle_deg))
    )
    inverse_cov = np.linalg.pinv(covariance, rcond=1e-8)
    weights_unscaled = inverse_cov @ steering
    denominator = np.vdot(steering, weights_unscaled)
    if abs(denominator) < 1e-12:
        raise ValueError("MVDR normalization nearly zero")
    weights = weights_unscaled / denominator

    # np.vdot(weights, x) = conjugate(weights) @ x.
    z_original = snapshots @ weights.conj()
    z_centered = centered @ weights.conj()

    if not np.isfinite(z_original).all() or np.median(np.abs(z_original)) <= 0:
        raise ValueError("Invalid or all-zero beamformed signal")

    def phase_features(z):
        wrapped = np.angle(z)
        unwrapped = np.unwrap(wrapped)
        # Linear detrending removes an arbitrary intercept and slow linear drift;
        # it is NOT proof that remaining fluctuations are physiological motion.
        dynamic = detrend(unwrapped, type="linear")
        incremental = np.diff(unwrapped, prepend=unwrapped[0])
        return wrapped, unwrapped, dynamic, incremental

    wrapped, unwrapped, dynamic, incremental = phase_features(z_original)
    centered_wrapped, centered_unwrapped, centered_dynamic, centered_incremental = (
        phase_features(z_centered)
    )
    median_amp = float(np.median(np.abs(z_original)))
    low_amp_pct = 100.0 * float(np.mean(np.abs(z_original) < 0.1 * median_amp))
    abrupt_steps = int(np.sum(np.abs(np.diff(unwrapped)) > 2.5))

    return {
        "range_bin": bin_index,
        "range_m": float(range_m[bin_index]),
        "angle_deg": angle_deg,
        "z": z_original,
        "z_mean_cancelled": z_centered,
        "phase_wrapped": wrapped,
        "phase_unwrapped": unwrapped,
        "phase_dynamic": dynamic,
        "phase_increment": incremental,
        "phase_mean_cancelled_dynamic": centered_dynamic,
        "phase_mean_cancelled_increment": centered_incremental,
        "low_amplitude_percent": low_amp_pct,
        "abrupt_steps": abrupt_steps,
        "weights": weights,
    }


def plot_phase_traces(data: list[dict], time_s: np.ndarray) -> Path:
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True, constrained_layout=True)
    for index, item in enumerate(data):
        heading = f"Candidate {index+1} ({item['angle_deg']:+.0f}°, {item['range_m']:.2f} m)"
        amplitude = np.abs(item["z"])
        axs[index, 0].plot(time_s, amplitude / np.median(amplitude), lw=0.8)
        axs[index, 0].set_ylabel(f"{heading}\nRelative |I/Q|")
        axs[index, 0].set_title("Original beamformed amplitude")
        axs[index, 1].plot(time_s, item["phase_wrapped"], lw=0.75)
        axs[index, 1].set_title("Wrapped phase")
        axs[index, 1].set_ylabel("Phase (rad)")
        axs[index, 2].plot(time_s, item["phase_dynamic"], lw=0.8)
        axs[index, 2].set_title("Unwrapped, linearly detrended phase")
        axs[index, 2].set_ylabel("Phase (rad)")
        for ax in axs[index]:
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=0.25)
    fig.suptitle("Exploratory complex I/Q and phase — no vital signs estimated", fontsize=13)
    dest = RESULTS_DIR / "phase_extraction.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


def plot_clutter_sensitivity(data: list[dict], time_s: np.ndarray) -> Path:
    """Compare the two phase preprocessing choices; no method declared superior."""
    fig, axs = plt.subplots(2, 2, figsize=(13, 7), sharex=True, constrained_layout=True)
    for index, item in enumerate(data):
        title = f"Candidate {index+1} ({item['angle_deg']:+.0f}°, {item['range_m']:.2f} m)"
        ax = axs[index, 0]
        ax.plot(time_s, item["phase_dynamic"], label="Original complex signal", lw=0.8)
        ax.plot(time_s, item["phase_mean_cancelled_dynamic"],
                label="Complex mean subtracted", lw=0.8, alpha=0.75)
        ax.set_title(title + " — unwrapped phase")
        ax.set_ylabel("Detrended phase (rad)")
        ax.legend(fontsize=8)
        ax = axs[index, 1]
        ax.plot(time_s, item["phase_increment"], label="Original", lw=0.7)
        ax.plot(time_s, item["phase_mean_cancelled_increment"],
                label="Mean cancelled", lw=0.7, alpha=0.65)
        ax.set_title(title + " — frame-to-frame phase change")
        ax.set_ylabel("Phase increment (rad/frame)")
        ax.legend(fontsize=8)
        for a in axs[index]:
            a.set_xlabel("Time (s)")
            a.grid(alpha=0.25)
    fig.suptitle("Sensitivity to slow-time complex mean cancellation (exploratory)", fontsize=13)
    dest = RESULTS_DIR / "phase_preprocessing_comparison.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


def main():
    if not RADAR_FILE.is_file():
        raise FileNotFoundError(f"Missing raw radar recording: {RADAR_FILE}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    radar = load_dca1000(RADAR_FILE)
    ranges, angles, mvdr = compute_range_angle_mvdr(radar)
    candidates, _ = find_two_candidates(ranges, angles, mvdr)
    if len(candidates) != 2:
        raise RuntimeError(f"Expected 2 candidate peaks, detected {len(candidates)}")

    range_fft = make_range_fft(radar)
    extracted = [beamform_candidate(range_fft, ranges, c) for c in candidates]
    time_s = np.arange(len(radar)) * FRAME_PERIOD_S

    print("=== Position 1 / 3 GHz: exploratory phase extraction ===")
    print(f"Input radar: {radar.shape}; duration: {len(time_s)*FRAME_PERIOD_S:.1f} s")
    for i, item in enumerate(extracted, start=1):
        print(f"Candidate {i}: angle={item['angle_deg']:+.0f}°, "
              f"range={item['range_m']:.2f} m, FFT bin={item['range_bin']}")
        print(f"  Phase SD after detrending: {np.std(item['phase_dynamic']):.2f} rad; "
              f"large frame increments (>2.5 rad): {item['abrupt_steps']}; "
              f"low-amplitude samples: {item['low_amplitude_percent']:.2f}%")

    # Keep derived measurements locally, because they depend on downloaded recordings.
    saved = PROCESSED_DIR / "candidate_phase_signals.npz"
    np.savez_compressed(
        saved,
        time_s=time_s,
        candidate_angles_deg=np.array([x["angle_deg"] for x in extracted]),
        candidate_ranges_m=np.array([x["range_m"] for x in extracted]),
        complex_iq=np.stack([x["z"] for x in extracted]),
        phase_unwrapped=np.stack([x["phase_unwrapped"] for x in extracted]),
        phase_dynamic=np.stack([x["phase_dynamic"] for x in extracted]),
        phase_increment=np.stack([x["phase_increment"] for x in extracted]),
        mean_cancelled_phase_dynamic=np.stack(
            [x["phase_mean_cancelled_dynamic"] for x in extracted]
        ),
    )
    print(f"Saved derived signals (local data): {saved}")
    print(f"Saved plot: {plot_phase_traces(extracted, time_s)}")
    print(f"Saved plot: {plot_clutter_sensitivity(extracted, time_s)}")
    print("CAUTION: Candidate identities and reference timing are not validated.")
    print("CAUTION: Detrended phase and its peaks are NOT yet confirmed vital signs.")


if __name__ == "__main__":
    main()
