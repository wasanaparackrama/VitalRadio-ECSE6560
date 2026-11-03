"""
Phase 2.5A - Simulated Human Chest Motion

Purpose:
    Generate a mathematical model of chest displacement
    caused by breathing and heartbeat.

Mathematical model:

    x(t) = A_b*cos(2*pi*f_b*t)
         + A_h*cos(2*pi*f_h*t)

This is a simulation-only model.
The values are known ground truth values that we will
later try to recover from the radar signal.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Slow-time sampling parameters
# ============================================================

# Number of FMCW chirps collected per second.
# This becomes the sampling rate of our slow-time signal.

slow_fs = 20              # 20 chirps/second

# Total observation time

duration = 60              # seconds

# Number of chirps

N = int(slow_fs * duration)


# ============================================================
# 2. Create slow-time vector
# ============================================================

t = np.arange(N) / slow_fs


# ============================================================
# 3. Breathing parameters
# ============================================================

breathing_frequency = 0.25     # Hz

breathing_amplitude = 0.002    # 2 mm


# ============================================================
# 4. Heartbeat parameters
# ============================================================

heart_frequency = 1.2          # Hz

heart_amplitude = 0.0002       # 0.2 mm


# ============================================================
# 5. Generate breathing motion
# ============================================================

breathing_motion = (
    breathing_amplitude
    * np.cos(
        2 * np.pi
        * breathing_frequency
        * t
    )
)


# ============================================================
# 6. Generate heartbeat motion
# ============================================================

heart_motion = (
    heart_amplitude
    * np.cos(
        2 * np.pi
        * heart_frequency
        * t
    )
)


# ============================================================
# 7. Total chest displacement
# ============================================================

chest_motion = (
    breathing_motion
    + heart_motion
)


# ============================================================
# 8. Print parameters
# ============================================================

print("==========================================")
print("PHASE 2.5A - CHEST MOTION")
print("==========================================")

print(f"Slow-time sampling rate : {slow_fs} Hz")
print(f"Observation time        : {duration} s")
print(f"Number of chirps        : {N}")

print("------------------------------------------")

print(
    f"Breathing frequency     : "
    f"{breathing_frequency:.2f} Hz"
)

print(
    f"Breathing rate          : "
    f"{breathing_frequency * 60:.1f} BPM"
)

print(
    f"Breathing amplitude     : "
    f"{breathing_amplitude * 1000:.2f} mm"
)

print("------------------------------------------")

print(
    f"Heart frequency         : "
    f"{heart_frequency:.2f} Hz"
)

print(
    f"Heart rate              : "
    f"{heart_frequency * 60:.1f} BPM"
)

print(
    f"Heartbeat amplitude     : "
    f"{heart_amplitude * 1000:.2f} mm"
)

print("==========================================")


# ============================================================
# 9. Plot breathing and heartbeat separately
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    t,
    breathing_motion * 1000,
    label="Breathing"
)

plt.plot(
    t,
    heart_motion * 1000,
    label="Heartbeat"
)

plt.xlabel("Time (s)")
plt.ylabel("Displacement (mm)")

plt.title("Simulated Chest Motion Components")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 10. Plot total chest displacement
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    t,
    chest_motion * 1000
)

plt.xlabel("Time (s)")
plt.ylabel("Displacement (mm)")

plt.title("Total Simulated Chest Displacement")

plt.grid(True)

plt.tight_layout()
plt.show()