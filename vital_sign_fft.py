"""
Phase 2.6 - Vital-Sign Frequency Estimation

Goal:
    Recover breathing rate and heart rate from the
    simulated radar phase signal.

Known simulation values:

    Breathing frequency = 0.25 Hz  -> 15 BPM
    Heart frequency     = 1.20 Hz  -> 72 BPM

The FFT should independently identify these frequencies.
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
# 3. Target
# ============================================================

R0 = 3.0


# ============================================================
# 4. Slow-time parameters
# ============================================================

slow_fs = 20              # chirps per second

duration = 60             # seconds

N = int(slow_fs * duration)

t = np.arange(N) / slow_fs


# ============================================================
# 5. Ground-truth vital-sign parameters
# ============================================================

fb_true = 0.25            # breathing frequency (Hz)

fh_true = 1.2             # heartbeat frequency (Hz)

Ab = 0.002                # breathing displacement (m)

Ah = 0.0002              # heartbeat displacement (m)


# ============================================================
# 6. Generate chest motion
# ============================================================

breathing = (
    Ab * np.cos(
        2 * np.pi * fb_true * t
    )
)

heartbeat = (
    Ah * np.cos(
        2 * np.pi * fh_true * t
    )
)

x = breathing + heartbeat


# ============================================================
# 7. Calculate range
# ============================================================

R = R0 + x


# ============================================================
# 8. Calculate radar phase
# ============================================================

phi = (
    4 * np.pi * R / wavelength
)


# ============================================================
# 9. Generate complex I/Q signal
# ============================================================

z = np.exp(1j * phi)


# ============================================================
# 10. Extract phase
# ============================================================

phase_wrapped = np.angle(z)

phase_unwrapped = np.unwrap(
    phase_wrapped
)


# ============================================================
# 11. Remove DC component
# ============================================================

phase_dynamic = (
    phase_unwrapped
    - np.mean(phase_unwrapped)
)


# ============================================================
# 12. FFT
# ============================================================

phase_fft = np.fft.rfft(
    phase_dynamic
)


# ============================================================
# 13. Frequency axis
# ============================================================

freq = np.fft.rfftfreq(
    N,
    d=1 / slow_fs
)


# ============================================================
# 14. Magnitude spectrum
# ============================================================

magnitude = (
    2 / N
    * np.abs(phase_fft)
)


# ============================================================
# 15. Ignore DC
# ============================================================

magnitude[0] = 0


# ============================================================
# 16. Search for breathing peak
# ============================================================

breathing_band = (
    (freq >= 0.1)
    &
    (freq <= 0.5)
)

breathing_indices = np.where(
    breathing_band
)[0]

breathing_peak_index = (
    breathing_indices[
        np.argmax(
            magnitude[breathing_indices]
        )
    ]
)

fb_estimated = freq[
    breathing_peak_index
]


# ============================================================
# 17. Search for heartbeat peak
# ============================================================

heart_band = (
    (freq >= 0.8)
    &
    (freq <= 2.0)
)

heart_indices = np.where(
    heart_band
)[0]

heart_peak_index = (
    heart_indices[
        np.argmax(
            magnitude[heart_indices]
        )
    ]
)

fh_estimated = freq[
    heart_peak_index
]


# ============================================================
# 18. Convert frequency to rates
# ============================================================

BR_estimated = (
    fb_estimated * 60
)

HR_estimated = (
    fh_estimated * 60
)


# ============================================================
# 19. Calculate errors
# ============================================================

BR_true = fb_true * 60

HR_true = fh_true * 60

BR_error = BR_estimated - BR_true

HR_error = HR_estimated - HR_true


# ============================================================
# 20. Print results
# ============================================================

print("==========================================")
print("PHASE 2.6 - VITAL SIGN FFT")
print("==========================================")

print("GROUND TRUTH")
print("------------------------------------------")

print(
    f"Breathing frequency : "
    f"{fb_true:.4f} Hz"
)

print(
    f"Breathing rate      : "
    f"{BR_true:.2f} BPM"
)

print(
    f"Heart frequency     : "
    f"{fh_true:.4f} Hz"
)

print(
    f"Heart rate          : "
    f"{HR_true:.2f} BPM"
)

print()
print("ESTIMATED")
print("------------------------------------------")

print(
    f"Breathing frequency : "
    f"{fb_estimated:.4f} Hz"
)

print(
    f"Breathing rate      : "
    f"{BR_estimated:.2f} BPM"
)

print(
    f"Heart frequency     : "
    f"{fh_estimated:.4f} Hz"
)

print(
    f"Heart rate          : "
    f"{HR_estimated:.2f} BPM"
)

print()
print("ERROR")
print("------------------------------------------")

print(
    f"Breathing error     : "
    f"{BR_error:.2f} BPM"
)

print(
    f"Heart-rate error    : "
    f"{HR_error:.2f} BPM"
)

print("==========================================")


# ============================================================
# 21. Plot phase spectrum
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    freq,
    magnitude
)

plt.axvline(
    fb_true,
    linestyle="--",
    label="True breathing frequency"
)

plt.axvline(
    fh_true,
    linestyle="--",
    label="True heart frequency"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase Spectrum Magnitude")

plt.title(
    "Radar Phase Spectrum - Vital Sign Detection"
)

plt.xlim(0, 2.5)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()