"""
============================================================
PHASE 2.9 - STATIC CLUTTER REMOVAL
============================================================

Goal:
    Remove a strong stationary reflection from the
    complex radar signal before extracting phase.

Signal model:

    z_received[n]
        =
    z_human[n]
        +
    z_static[n]

Since the static reflection is approximately constant:

    z_static_hat = mean(z_received)

Then:

    z_clean[n]
        =
    z_received[n] - z_static_hat

After clutter removal we extract the phase.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Slow-time parameters
# ============================================================

slow_fs = 20

duration = 60

N = int(
    slow_fs * duration
)

t = np.arange(N) / slow_fs


# ============================================================
# 2. Radar parameters
# ============================================================

c = 3e8

fc = 24e9

wavelength = c / fc


# ============================================================
# 3. Human parameters
# ============================================================

R0 = 3.0

fb = 0.25

fh = 1.20

Ab = 0.002

Ah = 0.0002


# ============================================================
# 4. Generate breathing motion
# ============================================================

breathing = (
    Ab
    * np.cos(
        2 * np.pi
        * fb
        * t
    )
)


# ============================================================
# 5. Generate heartbeat motion
# ============================================================

heartbeat = (
    Ah
    * np.cos(
        2 * np.pi
        * fh
        * t
    )
)


# ============================================================
# 6. Total chest displacement
# ============================================================

x = (
    breathing
    + heartbeat
)


# ============================================================
# 7. Human range
# ============================================================

R_human = (
    R0
    + x
)


# ============================================================
# 8. Human radar phase
# ============================================================

phi_human = (
    4 * np.pi
    * R_human
    / wavelength
)


# ============================================================
# 9. Human complex I/Q signal
# ============================================================

A_human = 1.0

z_human = (
    A_human
    * np.exp(
        1j * phi_human
    )
)


# ============================================================
# 10. Static reflection
# ============================================================

A_static = 2.0

phi_static = 1.0

z_static = (
    A_static
    * np.exp(
        1j * phi_static
    )
)


# Create a constant signal

z_static_signal = (
    np.ones(N)
    * z_static
)


# ============================================================
# 11. Received signal
# ============================================================

z_received = (
    z_human
    + z_static_signal
)


# ============================================================
# 12. Estimate static component
# ============================================================

z_static_estimated = (
    np.mean(z_received)
)


# ============================================================
# 13. Remove static component
# ============================================================

z_clean = (
    z_received
    - z_static_estimated
)


# ============================================================
# 14. Extract phase BEFORE clutter removal
# ============================================================

phase_before = np.unwrap(
    np.angle(
        z_received
    )
)

phase_before -= np.mean(
    phase_before
)


# ============================================================
# 15. Extract phase AFTER clutter removal
# ============================================================

phase_after = np.unwrap(
    np.angle(
        z_clean
    )
)

phase_after -= np.mean(
    phase_after
)


# ============================================================
# 16. Calculate FFT after clutter removal
# ============================================================

phase_fft = np.fft.rfft(
    phase_after
)

freq = np.fft.rfftfreq(
    N,
    d=1 / slow_fs
)

magnitude = (
    2 / N
    * np.abs(phase_fft)
)

magnitude[0] = 0


# ============================================================
# 17. Find breathing peak
# ============================================================

breathing_band = (
    (freq >= 0.10)
    &
    (freq <= 0.50)
)

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


# ============================================================
# 18. Find heartbeat peak
# ============================================================

heart_band = (
    (freq >= 0.80)
    &
    (freq <= 2.00)
)

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


# ============================================================
# 19. Convert to BPM
# ============================================================

BR_estimated = (
    fb_estimated * 60
)

HR_estimated = (
    fh_estimated * 60
)


# ============================================================
# 20. Print results
# ============================================================

print()
print("======================================================")
print("PHASE 2.9 - STATIC CLUTTER REMOVAL")
print("======================================================")

print(
    f"Static amplitude       : "
    f"{A_static:.2f}"
)

print(
    f"Human amplitude        : "
    f"{A_human:.2f}"
)

print(
    f"Static/Human ratio     : "
    f"{A_static / A_human:.2f}"
)

print("--------------------------------------")

print(
    f"Estimated breathing    : "
    f"{fb_estimated:.3f} Hz"
)

print(
    f"Estimated breathing    : "
    f"{BR_estimated:.2f} BPM"
)

print(
    f"Estimated heartbeat    : "
    f"{fh_estimated:.3f} Hz"
)

print(
    f"Estimated heart rate   : "
    f"{HR_estimated:.2f} BPM"
)

print("======================================================")


# ============================================================
# 21. Plot phase before clutter removal
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    t,
    phase_before
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Phase (rad)"
)

plt.title(
    "Phase Before Static Clutter Removal"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 22. Plot phase after clutter removal
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    t,
    phase_after
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Phase (rad)"
)

plt.title(
    "Phase After Static Clutter Removal"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 23. Plot spectrum after clutter removal
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    freq,
    magnitude
)

plt.axvline(
    fb,
    linestyle="--",
    label="True breathing"
)

plt.axvline(
    fh,
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
    "Vital-Sign Spectrum After Clutter Removal"
)

plt.xlim(
    0,
    2.5
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()