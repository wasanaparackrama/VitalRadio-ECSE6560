"""
============================================================
PHASE 2.10 - VITAL-SIGN BANDPASS FILTERING
============================================================

Goal:
    Separate breathing and heartbeat components from the
    radar phase signal.

Signal model:

    chest motion =
        breathing + heartbeat

Radar processing:

    I/Q
      ↓
    Phase
      ↓
    Static clutter removal
      ↓
    Bandpass filtering
      ↓
    Breathing / Heartbeat
      ↓
    FFT
      ↓
    BPM

Breathing band:
    0.1 - 0.5 Hz

Heartbeat band:
    0.8 - 2.0 Hz
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt


# ============================================================
# 1. Slow-time parameters
# ============================================================

slow_fs = 20.0

duration = 60.0

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
# 3. Ground-truth vital signs
# ============================================================

fb_true = 0.25

fh_true = 1.20


# ============================================================
# 4. Chest-motion amplitudes
# ============================================================

Ab = 0.002       # 2 mm

Ah = 0.0002      # 0.2 mm


# ============================================================
# 5. Target range
# ============================================================

R0 = 3.0


# ============================================================
# 6. Generate breathing motion
# ============================================================

breathing = (
    Ab *
    np.cos(
        2 * np.pi *
        fb_true *
        t
    )
)


# ============================================================
# 7. Generate heartbeat motion
# ============================================================

heartbeat = (
    Ah *
    np.cos(
        2 * np.pi *
        fh_true *
        t
    )
)


# ============================================================
# 8. Total chest displacement
# ============================================================

x = (
    breathing +
    heartbeat
)


# ============================================================
# 9. Human range
# ============================================================

R = (
    R0 +
    x
)


# ============================================================
# 10. Radar phase
# ============================================================

phi = (
    4 *
    np.pi *
    R /
    wavelength
)


# ============================================================
# 11. Generate complex radar signal
# ============================================================

A_human = 1.0

z_human = (
    A_human *
    np.exp(
        1j * phi
    )
)


# ============================================================
# 12. Add static reflection
# ============================================================

A_static = 2.0

phi_static = 1.0

z_static = (
    A_static *
    np.exp(
        1j * phi_static
    )
)

z_static_signal = (
    np.ones(N) *
    z_static
)


# ============================================================
# 13. Received signal
# ============================================================

z_received = (
    z_human +
    z_static_signal
)


# ============================================================
# 14. Static clutter removal
# ============================================================

static_estimate = np.mean(
    z_received
)

z_clean = (
    z_received -
    static_estimate
)


# ============================================================
# 15. Extract radar phase
# ============================================================

phase = np.angle(
    z_clean
)


# ============================================================
# 16. Unwrap phase
# ============================================================

phase_unwrapped = np.unwrap(
    phase
)


# ============================================================
# 17. Remove DC component
# ============================================================

phase_dynamic = (
    phase_unwrapped -
    np.mean(
        phase_unwrapped
    )
)


# ============================================================
# 18. Design breathing filter
# ============================================================

breathing_low = 0.10

breathing_high = 0.50

filter_order = 4

b_breathing, a_breathing = butter(
    filter_order,
    [
        breathing_low,
        breathing_high
    ],
    btype="bandpass",
    fs=slow_fs
)


# ============================================================
# 19. Design heartbeat filter
# ============================================================

heart_low = 0.80

heart_high = 2.00

b_heart, a_heart = butter(
    filter_order,
    [
        heart_low,
        heart_high
    ],
    btype="bandpass",
    fs=slow_fs
)


# ============================================================
# 20. Apply breathing filter
# ============================================================

breathing_signal = filtfilt(
    b_breathing,
    a_breathing,
    phase_dynamic
)


# ============================================================
# 21. Apply heartbeat filter
# ============================================================

heartbeat_signal = filtfilt(
    b_heart,
    a_heart,
    phase_dynamic
)


# ============================================================
# 22. FFT function
# ============================================================

def calculate_spectrum(signal_data):

    spectrum = np.fft.rfft(
        signal_data
    )

    magnitude = (
        2 / N *
        np.abs(spectrum)
    )

    frequency = np.fft.rfftfreq(
        N,
        d=1 / slow_fs
    )

    return frequency, magnitude


# ============================================================
# 23. Breathing spectrum
# ============================================================

freq_breathing, mag_breathing = (
    calculate_spectrum(
        breathing_signal
    )
)


# ============================================================
# 24. Heartbeat spectrum
# ============================================================

freq_heart, mag_heart = (
    calculate_spectrum(
        heartbeat_signal
    )
)


# ============================================================
# 25. Find breathing peak
# ============================================================

breathing_region = (
    (freq_breathing >= breathing_low)
    &
    (freq_breathing <= breathing_high)
)

breathing_indices = np.where(
    breathing_region
)[0]

breathing_peak_index = (
    breathing_indices[
        np.argmax(
            mag_breathing[
                breathing_indices
            ]
        )
    ]
)

fb_estimated = (
    freq_breathing[
        breathing_peak_index
    ]
)


# ============================================================
# 26. Find heartbeat peak
# ============================================================

heart_region = (
    (freq_heart >= heart_low)
    &
    (freq_heart <= heart_high)
)

heart_indices = np.where(
    heart_region
)[0]

heart_peak_index = (
    heart_indices[
        np.argmax(
            mag_heart[
                heart_indices
            ]
        )
    ]
)

fh_estimated = (
    freq_heart[
        heart_peak_index
    ]
)


# ============================================================
# 27. Convert to BPM
# ============================================================

BR_estimated = (
    fb_estimated * 60
)

HR_estimated = (
    fh_estimated * 60
)


# ============================================================
# 28. Calculate errors
# ============================================================

BR_error = (
    BR_estimated -
    fb_true * 60
)

HR_error = (
    HR_estimated -
    fh_true * 60
)


# ============================================================
# 29. Print results
# ============================================================

print()
print("======================================================")
print("PHASE 2.10 - VITAL-SIGN FILTERING")
print("======================================================")

print()
print("GROUND TRUTH")
print("--------------------------------------")

print(
    f"Breathing frequency : "
    f"{fb_true:.2f} Hz"
)

print(
    f"Breathing rate      : "
    f"{fb_true * 60:.2f} BPM"
)

print(
    f"Heartbeat frequency : "
    f"{fh_true:.2f} Hz"
)

print(
    f"Heart rate         : "
    f"{fh_true * 60:.2f} BPM"
)


print()
print("ESTIMATED")
print("--------------------------------------")

print(
    f"Breathing frequency : "
    f"{fb_estimated:.3f} Hz"
)

print(
    f"Breathing rate      : "
    f"{BR_estimated:.2f} BPM"
)

print(
    f"Heartbeat frequency : "
    f"{fh_estimated:.3f} Hz"
)

print(
    f"Heart rate         : "
    f"{HR_estimated:.2f} BPM"
)


print()
print("ERROR")
print("--------------------------------------")

print(
    f"Breathing error     : "
    f"{BR_error:.2f} BPM"
)

print(
    f"Heart-rate error    : "
    f"{HR_error:.2f} BPM"
)

print("======================================================")


# ============================================================
# 30. Plot original phase
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    t,
    phase_dynamic
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Phase (rad)"
)

plt.title(
    "Radar Phase Before Vital-Sign Filtering"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 31. Plot breathing component
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    t,
    breathing_signal
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Phase (rad)"
)

plt.title(
    "Extracted Breathing Component"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 32. Plot heartbeat component
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    t,
    heartbeat_signal
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Phase (rad)"
)

plt.title(
    "Extracted Heartbeat Component"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 33. Plot breathing spectrum
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    freq_breathing,
    mag_breathing
)

plt.axvline(
    fb_true,
    linestyle="--",
    label="True breathing"
)

plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Magnitude"
)

plt.title(
    "Breathing Spectrum"
)

plt.xlim(
    0,
    0.8
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 34. Plot heartbeat spectrum
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    freq_heart,
    mag_heart
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
    "Heartbeat Spectrum"
)

plt.xlim(
    0.5,
    2.5
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()