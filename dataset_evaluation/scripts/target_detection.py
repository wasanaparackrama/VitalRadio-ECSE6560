"""Exploratory two-person localization using range-angle MVDR beamforming.

Dataset: Mendeley, FMCW radar-based multi-person vital sign monitoring data.
Recording: adc_3GHZ_position1_ (1).bin (60 GHz sensor, 3 GHz bandwidth).

Run from the repository root:
    python dataset_evaluation/scripts/target_detection.py

Requirements: numpy, scipy, matplotlib, and load_radar_data.py in this folder.
Output: dataset_evaluation/results/target_detection.png

This script estimates *candidate* target locations from the 8 virtual
channels selected by the authors. A peak is not proof of a person's identity.
The sign of the azimuth angle depends on array orientation/convention;
we do not use it to assign Target1 / Target2 CSV identities yet.

Approach follows the dataset authors' MTI.m, IWR6843ISK_DOA.m, and Main.m:
  slow-time mean cancellation -> range FFT -> covariance per range bin
  -> MVDR angle spectrum -> spatial peak detection.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from load_radar_data import (
    ADC_RATE_HZ,
    ADC_SAMPLES,
    CHIRP_SLOPE_HZ_PER_S,
    RADAR_FILE,
    SPEED_OF_LIGHT,
    load_dca1000,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
NFFT = 256
ANGLES_DEG = np.arange(-60.0, 61.0, 1.0)
SHOW_MAX_RANGE_M = 2.0
SEARCH_MIN_RANGE_M = 0.6
SEARCH_MAX_RANGE_M = 1.6
MIN_ANGLE_SEPARATION_DEG = 15
DIAGONAL_LOADING = 0.01  # improve numerical stability of covariance inverse


def compute_range_angle_mvdr(radar):
    """Return (range_m, angle_deg, spectrum) for range-angle power (linear)."""
    frames, adc_samples, antennas = radar.shape
    if antennas != 8:
        raise ValueError(f"This 8-element steering model requires 8 channels, got {antennas}")
    if adc_samples != ADC_SAMPLES:
        raise ValueError(f"Expected {ADC_SAMPLES} ADC samples, got {adc_samples}")

    # Remove frame-constant contribution at each ADC sample & antenna.
    # For localization only; phase estimation later needs separate validation.
    moving = radar - radar.mean(axis=0, keepdims=True)
    window = np.hanning(adc_samples)[None, :, None]
    range_fft = np.fft.fft(moving * window, n=NFFT, axis=1)[:, :NFFT // 2, :]

    # Frequency -> distance via the FMCW range equation.
    frequency_hz = np.arange(NFFT // 2) * ADC_RATE_HZ / NFFT
    ranges_m = SPEED_OF_LIGHT * frequency_hz / (2.0 * CHIRP_SLOPE_HZ_PER_S)

    # Spatial covariance at each range: R[r] = E[x[r] x[r]^H].
    covariance = np.einsum(
        "fri,frj->rij", range_fft, range_fft.conj(), optimize=True
    ) / frames

    # Add diagonal loading to reduce instability for near-singular matrices.
    channel_power = np.trace(covariance, axis1=1, axis2=2).real / antennas
    covariance += (
        (DIAGONAL_LOADING * channel_power + 1e-8)[:, None, None]
        * np.eye(antennas)[None, :, :]
    )
    inverse_cov = np.linalg.pinv(covariance, rcond=1e-8)

    # Half-wavelength-spaced, 8-element virtual ULA, following the supplied
    # MATLAB DOA code. Angles are relative to array broadside.
    steering = np.exp(
        -1j * np.pi * np.arange(antennas)[:, None]
        * np.sin(np.deg2rad(ANGLES_DEG))[None, :]
    )
    denominator = np.einsum(
        "ia,rij,ja->ra", steering.conj(), inverse_cov, steering, optimize=True
    ).real
    mvdr = 1.0 / np.maximum(denominator, 1e-12)
    return ranges_m, ANGLES_DEG, mvdr


def find_two_candidates(ranges_m, angles_deg, mvdr):
    """Find two separated peaks in the documented near-1m subject region."""
    # Small display/detection smoothing reduces isolated spurious peaks.
    smooth = gaussian_filter(mvdr, sigma=(1.2, 1.8))
    search_range = (ranges_m >= SEARCH_MIN_RANGE_M) & (ranges_m <= SEARCH_MAX_RANGE_M)
    local_max = maximum_filter(smooth, size=(9, 19), mode="nearest")
    maxima = np.argwhere((smooth == local_max) & search_range[:, None])
    maxima = sorted(maxima, key=lambda ij: smooth[tuple(ij)], reverse=True)

    chosen = []
    for r_idx, a_idx in maxima:
        proposed = float(angles_deg[a_idx])
        if all(abs(proposed - old["angle_deg"]) >= MIN_ANGLE_SEPARATION_DEG for old in chosen):
            chosen.append({
                "range_m": float(ranges_m[r_idx]),
                "angle_deg": proposed,
                "relative_strength": float(smooth[r_idx, a_idx] / max(smooth.max(), 1e-12)),
            })
        if len(chosen) == 2:
            break
    return chosen, smooth


def plot_results(ranges_m, angles_deg, mvdr, candidates, smoothed):
    reference = max(smoothed.max(), 1e-12)
    db = 10.0 * np.log10(np.maximum(mvdr / reference, 1e-12))
    display_mask = ranges_m <= SHOW_MAX_RANGE_M
    near_mask = (ranges_m >= SEARCH_MIN_RANGE_M) & (ranges_m <= SEARCH_MAX_RANGE_M)
    angle_curve = smoothed[near_mask, :].max(axis=0) / reference

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True
    )
    im = ax.pcolormesh(
        angles_deg, ranges_m[display_mask], db[display_mask, :],
        shading="nearest", cmap="viridis", vmin=-40, vmax=0,
    )
    ax.axhline(1.0, ls="--", color="white", alpha=0.65, lw=1, label="Nominal 1 m distance")
    for i, candidate in enumerate(candidates, start=1):
        angle = candidate["angle_deg"]
        distance = candidate["range_m"]
        ax.plot(angle, distance, marker="x", color="red", ms=12, mew=2.8)
        label = f"Candidate {i}: {angle:+.0f}°, {distance:.2f} m"
        dx = -3 if angle > 0 else 3
        ha = "right" if angle > 0 else "left"
        ax.annotate(label, (angle, distance), xytext=(dx, 14),
                    textcoords="offset points", ha=ha, va="bottom",
                    fontsize=10, weight="bold", color="white",
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=3))
    ax.set_ylim(0, SHOW_MAX_RANGE_M)
    ax.set_xlim(angles_deg.min(), angles_deg.max())
    ax.set_xlabel("Estimated azimuth angle (degrees; sign convention unverified)")
    ax.set_ylabel("Estimated range (m)")
    ax.set_title("Position 1: MVDR range-angle spectrum — 8 virtual radar channels")
    ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ax, label="Relative MVDR spatial power (dB)")

    ax2.plot(angles_deg, 10*np.log10(np.maximum(angle_curve, 1e-12)), lw=1.8)
    for i, candidate in enumerate(candidates, start=1):
        ax2.axvline(candidate["angle_deg"], linestyle="--", lw=1.2, label=f"Candidate {i}")
    ax2.set_xlim(-60, 60)
    ax2.set_xlabel("Estimated azimuth angle (degrees)")
    ax2.set_ylabel("Relative response (dB)")
    ax2.set_title("Strongest MVDR response per angle (0.6–1.6 m range window)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RESULTS_DIR / "target_detection.png"
    fig.savefig(dest, dpi=160)
    print(f"Saved plot: {dest}")
    plt.show()


def main():
    if not RADAR_FILE.is_file():
        raise FileNotFoundError(f"Missing radar file: {RADAR_FILE}")
    radar = load_dca1000(RADAR_FILE)
    print(f"Loaded: {radar.shape} (frames, ADC samples, channels)")
    ranges, angles, power = compute_range_angle_mvdr(radar)
    candidates, smooth = find_two_candidates(ranges, angles, power)
    print("\nExploratory MVDR target candidates:")
    for i, p in enumerate(candidates, 1):
        print(f"  Candidate {i}: angle {p['angle_deg']:+.0f}°, "
              f"range {p['range_m']:.2f} m, "
              f"relative response {p['relative_strength']:.2f}")
    if len(candidates) < 2:
        print("WARNING: Two well-separated peaks were not found.")
    print("Dataset Position 1 reference: 2 subjects at 1.0 m, approx. 20° / 40° on opposing sides.")
    print("NOTE: Candidate numbering does NOT assign Target1/Target2 identity or validate vital signs.")
    print("NOTE: Range offsets and angular sign conventions require calibration/verification.")
    plot_results(ranges, angles, power, candidates, smooth)


if __name__ == "__main__":
    main()
