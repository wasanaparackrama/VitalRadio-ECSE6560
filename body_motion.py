"""
============================================================
PHASE 2.11 - BODY MOTION / LARGE MOTION INTERFERENCE
============================================================

Goal:
    Investigate how large body movement affects radar-based
    breathing and heartbeat estimation.

New signal model:

    x(t) =
        breathing
        + heartbeat
        + body movement

We first use a slow body movement at 0.05 Hz.

The body movement is intentionally much larger than the
heartbeat displacement.

Ground truth:

    Breathing = 0.25 Hz = 15 BPM
    Heartbeat = 1.20 Hz = 72 BPM
    Body motion = 0.05 Hz = 3 movements/min
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
# 3. Target range
# ============================================================

R0 = 3.0


# ============================================================
# 4. Vital-sign frequencies
# ============================================================

fb = 0.25

fh = 1.20


# ============================================================
# 5. Body-motion frequency
# ============================================================

fm = 0.05


# ============================================================
# 6. Displacement amplitudes
# ============================================================

# Breathing = 2 mm

Ab = 0.002


# Heartbeat = 0.2 mm

Ah = 0.0002


# Body movement = 20 mm

Am = 0.020


# ============================================================
# 7. Breathing displacement
# ============================================================

breathing = (
    Ab *
    np.cos(
        2 * np.pi *
        fb *
        t
    )
)


# ============================================================
# 8. Heartbeat displacement
# ============================================================

heartbeat = (
    Ah *
    np.cos(
        2 * np.pi *
        fh *
        t
    )
)


# ============================================================
# 9. Large body movement
# ============================================================

body_motion = (
    Am *
    np.cos(
        2 * np.pi *
        fm *
        t
    )
)


# ============================================================
# 10. Total chest/body displacement
# ============================================================

x = (
    breathing
    + heartbeat
    + body_motion
)


# ============================================================
# 11. Human range
# ============================================================

R = (
    R0 +
    x
)


# ============================================================
# 12. Radar phase
# ============================================================

phi = (
    4 *
    np.pi *
    R /
    wavelength
)


# ============================================================
# 13. Complex radar signal
# ============================================================

z_human = np.exp(
    1j * phi
)


# ============================================================
# 14. Extract phase
# ============================================================

phase = np.angle(
    z_human
)


# ============================================================
# 15. Unwrap phase
# ============================================================

phase_unwrapped = np.unwrap(
    phase
)


# ============================================================
# 16. Remove DC
# ============================================================

phase_dynamic = (
    phase_unwrapped -
    np.mean(
        phase_unwrapped
    )
)


# ============================================================
# 17. Design breathing filter
# ============================================================

breathing_low = 0.10

breathing_high = 0.50

order = 4

b_breathing, a_breathing = butter(
    order,
    [
        breathing_low,
        breathing_high
    ],
    btype="bandpass",
    fs=slow_fs
)


# ============================================================
# 18. Design heartbeat filter
# ============================================================

heart_low = 0.80

heart_high = 2.00

b_heart, a_heart = butter(
    order,
    [
        heart_low,
        heart_high
    ],
    btype="bandpass",
    fs=slow_fs
)


# ============================================================
# 19. Apply breathing filter
# ============================================================

breathing_filtered = filtfilt(
    b_breathing,
    a_breathing,
    phase_dynamic
)


# ============================================================
# 20. Apply heartbeat filter
# ============================================================

heartbeat_filtered = filtfilt(
    b_heart,
    a_heart,
    phase_dynamic
)


# ============================================================
# 21. FFT helper
# ============================================================

def calculate_fft(signal_data):

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
# 22. FFT of original phase
# ============================================================

freq_original, mag_original = (
    calculate_fft(
        phase_dynamic
    )
)


# ============================================================
# 23. FFT of breathing component
# ============================================================

freq_breathing, mag_breathing = (
    calculate_fft(
        breathing_filtered
    )
)


# ============================================================
# 24. FFT of heartbeat component
# ============================================================

freq_heart, mag_heart = (
    calculate_fft(
        heartbeat_filtered
    )
)


# ============================================================
# 25. Estimate breathing frequency
# ============================================================

breathing_region = (
    (freq_breathing >= 0.10)
    &
    (freq_breathing <= 0.50)
)

breathing_indices = np.where(
    breathing_region
)[0]

breathing_peak = (
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
        breathing_peak
    ]
)


# ============================================================
# 26. Estimate heartbeat frequency
# ============================================================

heart_region = (
    (freq_heart >= 0.80)
    &
    (freq_heart <= 2.00)
)

heart_indices = np.where(
    heart_region
)[0]

heart_peak = (
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
        heart_peak
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
    fb * 60
)

HR_error = (
    HR_estimated -
    fh * 60
)


# ============================================================
# 29. Print results
# ============================================================

print()
print("======================================================")
print("PHASE 2.11 - BODY MOTION ANALYSIS")
print("======================================================")

print()
print("GROUND TRUTH")
print("--------------------------------------")

print(
    f"Breathing frequency : "
    f"{fb:.2f} Hz"
)

print(
    f"Breathing rate      : "
    f"{fb * 60:.2f} BPM"
)

print(
    f"Heartbeat frequency : "
    f"{fh:.2f} Hz"
)

print(
    f"Heart rate          : "
    f"{fh * 60:.2f} BPM"
)

print(
    f"Body-motion frequency: "
    f"{fm:.2f} Hz"
)

print(
    f"Body motion rate    : "
    f"{fm * 60:.2f} /min"
)


print()
print("AMPLITUDES")
print("--------------------------------------")

print(
    f"Breathing amplitude : "
    f"{Ab * 1000:.2f} mm"
)

print(
    f"Heartbeat amplitude : "
    f"{Ah * 1000:.2f} mm"
)

print(
    f"Body-motion amplitude: "
    f"{Am * 1000:.2f} mm"
)


print()
print("ESTIMATION")
print("--------------------------------------")

print(
    f"Estimated breathing : "
    f"{fb_estimated:.3f} Hz"
)

print(
    f"Estimated BR        : "
    f"{BR_estimated:.2f} BPM"
)

print(
    f"Estimated heartbeat : "
    f"{fh_estimated:.3f} Hz"
)

print(
    f"Estimated HR        : "
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
# 30. Plot individual motion components
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    t,
    breathing * 1000,
    label="Breathing"
)

plt.plot(
    t,
    heartbeat * 1000,
    label="Heartbeat"
)

plt.plot(
    t,
    body_motion * 1000,
    label="Body Motion"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Displacement (mm)"
)

plt.title(
    "Vital Signs and Large Body Motion"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 31. Plot total displacement
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    t,
    x * 1000
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Displacement (mm)"
)

plt.title(
    "Total Human Displacement"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 32. Plot original phase
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
    "Radar Phase With Large Body Motion"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 33. Plot original phase spectrum
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    freq_original,
    mag_original
)

plt.axvline(
    fm,
    linestyle="--",
    label="Body motion"
)

plt.axvline(
    fb,
    linestyle="--",
    label="Breathing"
)

plt.axvline(
    fh,
    linestyle="--",
    label="Heartbeat"
)

plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Magnitude"
)

plt.title(
    "Radar Phase Spectrum With Body Motion"
)

plt.xlim(
    0,
    2.5
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 34. Plot filtered breathing signal
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    t,
    breathing_filtered
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Phase (rad)"
)

plt.title(
    "Breathing Component After Filtering"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 35. Plot filtered heartbeat signal
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    t,
    heartbeat_filtered
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Phase (rad)"
)

plt.title(
    "Heartbeat Component After Filtering"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 36. Plot breathing spectrum
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    freq_breathing,
    mag_breathing
)

plt.axvline(
    fb,
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
    "Breathing Spectrum With Body Motion"
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
# 37. Plot heartbeat spectrum
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    freq_heart,
    mag_heart
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
    "Heartbeat Spectrum With Body Motion"
)

plt.xlim(
    0.5,
    2.5
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()