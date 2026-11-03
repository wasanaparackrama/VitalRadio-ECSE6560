"""
Phase 2.5B - Convert Chest Motion to Radar Phase

Mathematical model:

    R(t) = R0 + x(t)

    lambda = c / fc

    phi(t) = 4*pi*R(t)/lambda

    z(t) = A*exp(j*phi(t))

This represents the complex I/Q signal observed at
the selected human range bin.
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

fc = 24e9                  # 24 GHz simulation example

wavelength = c / fc


# ============================================================
# 3. Target average range
# ============================================================

R0 = 3.0                   # 3 meters


# ============================================================
# 4. Slow-time parameters
# ============================================================

slow_fs = 20               # 20 chirps/second

duration = 60              # 60 seconds

N = int(slow_fs * duration)

t = np.arange(N) / slow_fs


# ============================================================
# 5. Breathing parameters
# ============================================================

fb = 0.25                  # 0.25 Hz = 15 BPM

Ab = 0.002                 # 2 mm


# ============================================================
# 6. Heartbeat parameters
# ============================================================

fh = 1.2                   # 1.2 Hz = 72 BPM

Ah = 0.0002                # 0.2 mm


# ============================================================
# 7. Generate chest motion
# ============================================================

x_breathing = (
    Ab * np.cos(2 * np.pi * fb * t)
)

x_heartbeat = (
    Ah * np.cos(2 * np.pi * fh * t)
)

x = x_breathing + x_heartbeat


# ============================================================
# 8. Calculate instantaneous range
# ============================================================

R = R0 + x


# ============================================================
# 9. Calculate radar phase
# ============================================================

phi = (
    4 * np.pi * R / wavelength
)


# ============================================================
# 10. Generate complex I/Q signal
# ============================================================

A = 1.0

z = A * np.exp(1j * phi)


# ============================================================
# 11. Extract I and Q
# ============================================================

I = np.real(z)

Q = np.imag(z)


# ============================================================
# 12. Extract wrapped phase
# ============================================================

phase_wrapped = np.angle(z)


# ============================================================
# 13. Unwrap phase
# ============================================================

phase_unwrapped = np.unwrap(
    phase_wrapped
)


# ============================================================
# 14. Remove constant phase component
# ============================================================

phase_dynamic = (
    phase_unwrapped
    - np.mean(phase_unwrapped)
)


# ============================================================
# 15. Print information
# ============================================================

print("==========================================")
print("PHASE 2.5B - RADAR PHASE MODEL")
print("==========================================")

print(f"Carrier frequency : {fc/1e9:.1f} GHz")
print(f"Wavelength        : {wavelength*1000:.2f} mm")
print(f"Average range     : {R0:.2f} m")

print("------------------------------------------")

print(f"Breathing rate    : {fb*60:.1f} BPM")
print(f"Heart rate        : {fh*60:.1f} BPM")

print("==========================================")


# ============================================================
# 16. Plot range variation
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    t,
    R
)

plt.xlabel("Time (s)")
plt.ylabel("Range (m)")

plt.title(
    "Simulated Human Range Variation"
)

plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 17. Plot wrapped phase
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    t,
    phase_wrapped
)

plt.xlabel("Time (s)")
plt.ylabel("Phase (rad)")

plt.title(
    "Wrapped Radar Phase"
)

plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 18. Plot unwrapped phase
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    t,
    phase_dynamic
)

plt.xlabel("Time (s)")
plt.ylabel("Phase (rad)")

plt.title(
    "Dynamic Unwrapped Radar Phase"
)

plt.grid(True)

plt.tight_layout()
plt.show()