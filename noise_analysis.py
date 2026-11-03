"""
============================================================
PHASE 2.7 - NOISE ANALYSIS
============================================================

Purpose:
    Add complex Gaussian noise to the simulated radar
    I/Q signal and investigate how noise affects:

        1. Breathing-rate estimation
        2. Heart-rate estimation
        3. Detection accuracy

Signal model:

    z_clean[n] = A * exp(j*phi[n])

    z_noisy[n] = z_clean[n] + w[n]

where:

    w[n] ~ Complex Gaussian Noise

The ground-truth values are:

    Breathing = 0.25 Hz = 15 BPM
    Heart     = 1.20 Hz = 72 BPM
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Constants
# ============================================================

c = 3e8


# ============================================================
# 2. Radar parameters
# ============================================================

fc = 24e9

wavelength = c / fc


# ============================================================
# 3. Target parameters
# ============================================================

R0 = 3.0


# ============================================================
# 4. Slow-time parameters
# ============================================================

slow_fs = 20

duration = 60

N = int(slow_fs * duration)

t = np.arange(N) / slow_fs


# ============================================================
# 5. Ground-truth vital signs
# ============================================================

fb_true = 0.25          # Hz

fh_true = 1.20          # Hz

Ab = 0.002              # 2 mm

Ah = 0.0002             # 0.2 mm


# ============================================================
# 6. Generate chest motion
# ============================================================

breathing = (
    Ab *
    np.cos(
        2 * np.pi *
        fb_true *
        t
    )
)

heartbeat = (
    Ah *
    np.cos(
        2 * np.pi *
        fh_true *
        t
    )
)

x = breathing + heartbeat


# ============================================================
# 7. Calculate target range
# ============================================================

R = R0 + x


# ============================================================
# 8. Calculate radar phase
# ============================================================

phi = (
    4 * np.pi *
    R /
    wavelength
)


# ============================================================
# 9. Generate clean complex I/Q signal
# ============================================================

A = 1.0

z_clean = (
    A *
    np.exp(1j * phi)
)


# ============================================================
# 10. Define FFT frequency axis
# ============================================================

freq = np.fft.rfftfreq(
    N,
    d=1 / slow_fs
)


# ============================================================
# 11. Define vital-sign search bands
# ============================================================

breathing_band = (
    (freq >= 0.10) &
    (freq <= 0.50)
)

heart_band = (
    (freq >= 0.80) &
    (freq <= 2.00)
)


# ============================================================
# 12. Function for vital-sign estimation
# ============================================================

def estimate_vital_signs(z):
    """
    Extract phase and estimate breathing and heart rates.
    """

    # --------------------------------------------------------
    # Extract wrapped phase
    # --------------------------------------------------------

    phase_wrapped = np.angle(z)

    # --------------------------------------------------------
    # Unwrap phase
    # --------------------------------------------------------

    phase_unwrapped = np.unwrap(
        phase_wrapped
    )

    # --------------------------------------------------------
    # Remove DC component
    # --------------------------------------------------------

    phase_dynamic = (
        phase_unwrapped -
        np.mean(phase_unwrapped)
    )

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    phase_fft = np.fft.rfft(
        phase_dynamic
    )

    magnitude = (
        2 / N *
        np.abs(phase_fft)
    )

    # Ignore DC
    magnitude[0] = 0

    # --------------------------------------------------------
    # Breathing peak
    # --------------------------------------------------------

    breathing_indices = np.where(
        breathing_band
    )[0]

    breathing_peak = (
        breathing_indices[
            np.argmax(
                magnitude[
                    breathing_indices
                ]
            )
        ]
    )

    fb_estimated = freq[
        breathing_peak
    ]

    # --------------------------------------------------------
    # Heartbeat peak
    # --------------------------------------------------------

    heart_indices = np.where(
        heart_band
    )[0]

    heart_peak = (
        heart_indices[
            np.argmax(
                magnitude[
                    heart_indices
                ]
            )
        ]
    )

    fh_estimated = freq[
        heart_peak
    ]

    # --------------------------------------------------------
    # Convert frequency to BPM
    # --------------------------------------------------------

    BR = fb_estimated * 60

    HR = fh_estimated * 60

    return (
        BR,
        HR,
        phase_dynamic,
        magnitude
    )


# ============================================================
# 13. SNR values to test
# ============================================================

snr_values = [
    30,
    20,
    10,
    5,
    0
]


# ============================================================
# 14. Store results
# ============================================================

results = []


# ============================================================
# 15. Random seed
# ============================================================

# This makes the experiment reproducible.

np.random.seed(42)


# ============================================================
# 16. Run noise experiments
# ============================================================

for snr_db in snr_values:

    # --------------------------------------------------------
    # Signal power
    # --------------------------------------------------------

    signal_power = np.mean(
        np.abs(z_clean) ** 2
    )

    # --------------------------------------------------------
    # Convert SNR from dB to linear
    # --------------------------------------------------------

    snr_linear = (
        10 ** (snr_db / 10)
    )

    # --------------------------------------------------------
    # Calculate required noise power
    # --------------------------------------------------------

    noise_power = (
        signal_power /
        snr_linear
    )

    # --------------------------------------------------------
    # Generate complex Gaussian noise
    # --------------------------------------------------------

    noise = (
        np.sqrt(
            noise_power / 2
        )
        *
        (
            np.random.randn(N)
            +
            1j *
            np.random.randn(N)
        )
    )

    # --------------------------------------------------------
    # Add noise
    # --------------------------------------------------------

    z_noisy = (
        z_clean +
        noise
    )

    # --------------------------------------------------------
    # Estimate vital signs
    # --------------------------------------------------------

    (
        BR_estimated,
        HR_estimated,
        phase_dynamic,
        magnitude
    ) = estimate_vital_signs(
        z_noisy
    )

    # --------------------------------------------------------
    # Calculate errors
    # --------------------------------------------------------

    BR_error = (
        BR_estimated -
        fb_true * 60
    )

    HR_error = (
        HR_estimated -
        fh_true * 60
    )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append(
        [
            snr_db,
            BR_estimated,
            HR_estimated,
            BR_error,
            HR_error
        ]
    )


# ============================================================
# 17. Print results
# ============================================================

print()
print("==========================================================")
print("PHASE 2.7 - NOISE ANALYSIS RESULTS")
print("==========================================================")

print(
    f"{'SNR (dB)':<12}"
    f"{'BR (BPM)':<15}"
    f"{'HR (BPM)':<15}"
    f"{'BR Error':<15}"
    f"{'HR Error':<15}"
)

print("-" * 72)

for result in results:

    snr_db = result[0]

    BR = result[1]

    HR = result[2]

    BR_error = result[3]

    HR_error = result[4]

    print(
        f"{snr_db:<12}"
        f"{BR:<15.2f}"
        f"{HR:<15.2f}"
        f"{BR_error:<15.2f}"
        f"{HR_error:<15.2f}"
    )

print("==========================================================")

print()
print("Ground truth:")
print(f"Breathing rate = {fb_true * 60:.2f} BPM")
print(f"Heart rate     = {fh_true * 60:.2f} BPM")


# ============================================================
# 18. Plot example spectra
# ============================================================

# We will show two cases:
#     20 dB
#     0 dB


plot_snr_values = [20, 0]


for snr_db in plot_snr_values:

    # --------------------------------------------------------
    # Calculate noise
    # --------------------------------------------------------

    signal_power = np.mean(
        np.abs(z_clean) ** 2
    )

    snr_linear = (
        10 ** (snr_db / 10)
    )

    noise_power = (
        signal_power /
        snr_linear
    )

    # --------------------------------------------------------
    # Generate noise
    # --------------------------------------------------------

    noise = (
        np.sqrt(
            noise_power / 2
        )
        *
        (
            np.random.randn(N)
            +
            1j *
            np.random.randn(N)
        )
    )

    # --------------------------------------------------------
    # Noisy signal
    # --------------------------------------------------------

    z_noisy = (
        z_clean +
        noise
    )

    # --------------------------------------------------------
    # Estimate
    # --------------------------------------------------------

    (
        BR,
        HR,
        phase_dynamic,
        magnitude
    ) = estimate_vital_signs(
        z_noisy
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        freq,
        magnitude
    )

    plt.axvline(
        fb_true,
        linestyle="--",
        label="True breathing"
    )

    plt.axvline(
        fh_true,
        linestyle="--",
        label="True heartbeat"
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Magnitude"
    )

    plt.title(
        f"Vital-Sign Spectrum "
        f"with {snr_db} dB SNR"
    )

    plt.xlim(
        0,
        2.5
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()