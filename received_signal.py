"""
Phase 2.2 - Simulating the Received FMCW Signal

Purpose:
    Model the reflection from a stationary human target.

Mathematical model:

    tau = 2*R0/c

    s_rx(t) = alpha * s_tx(t - tau)

At this stage:
    - The target is stationary.
    - No breathing is included.
    - No heartbeat is included.
    - No noise is included yet.
    - We are only modeling propagation delay.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Physical constants
# ============================================================

c = 3e8                  # Speed of light (m/s)


# ============================================================
# 2. FMCW simulation parameters
# ============================================================

A = 1.0                  # Transmitted amplitude

f0 = 1e6                 # Starting frequency = 1 MHz

B = 100e6                # Sweep bandwidth = 100 MHz

Tc = 1e-3                # Chirp duration = 1 ms

S = B / Tc               # Chirp slope


# ============================================================
# 3. Target parameters
# ============================================================

R0 = 3.0                 # Target range = 3 meters

alpha = 0.8              # Reflection/attenuation coefficient


# ============================================================
# 4. Sampling parameters
# ============================================================

fs = 250e6               # Sampling frequency = 250 MHz

N = int(Tc * fs)


# ============================================================
# 5. Create time vector
# ============================================================

t = np.arange(N) / fs


# ============================================================
# 6. Calculate propagation delay
# ============================================================

tau = (2 * R0) / c


# ============================================================
# 7. Calculate equivalent delay in samples
# ============================================================

delay_samples = int(round(tau * fs))


# ============================================================
# 8. Generate transmitted FMCW signal
# ============================================================

phase_tx = 2 * np.pi * (
    f0 * t + (S / 2) * t**2
)

s_tx = A * np.exp(1j * phase_tx)


# ============================================================
# 9. Generate delayed received signal
# ============================================================

s_rx = np.zeros_like(s_tx)

s_rx[delay_samples:] = (
    alpha * s_tx[:-delay_samples]
)


# ============================================================
# 10. Calculate instantaneous frequencies
# ============================================================

f_tx = f0 + S * t

f_rx = np.zeros_like(t)

valid = t >= tau

f_rx[valid] = f0 + S * (t[valid] - tau)


# ============================================================
# 11. Calculate theoretical beat frequency
# ============================================================

f_beat_theoretical = S * tau


# ============================================================
# 12. Display important information
# ============================================================

print("==========================================")
print("PHASE 2.2 - RECEIVED SIGNAL")
print("==========================================")

print(f"Target range        : {R0:.2f} m")
print(f"Reflection factor   : {alpha:.2f}")

print("------------------------------------------")

print(f"Chirp starting freq : {f0/1e6:.2f} MHz")
print(f"Chirp bandwidth     : {B/1e6:.2f} MHz")
print(f"Chirp duration      : {Tc*1e3:.2f} ms")
print(f"Chirp slope         : {S/1e9:.2f} GHz/s")

print("------------------------------------------")

print(f"Round-trip delay    : {tau*1e9:.2f} ns")
print(f"Delay in samples    : {delay_samples}")

print("------------------------------------------")

print(
    f"Theoretical beat frequency : "
    f"{f_beat_theoretical/1e3:.2f} kHz"
)

print("==========================================")


# ============================================================
# 13. Plot transmitted and received signals
#     Zoom into the beginning of the chirp
# ============================================================

zoom_time = 5e-6

zoom = t <= zoom_time


plt.figure(figsize=(10, 5))

plt.plot(
    t[zoom] * 1e6,
    np.real(s_tx[zoom]),
    label="Transmitted"
)

plt.plot(
    t[zoom] * 1e6,
    np.real(s_rx[zoom]),
    label="Received"
)

plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")

plt.title(
    "Transmitted and Delayed Received FMCW Signals"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 14. Plot the instantaneous frequency
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    t * 1e3,
    f_tx / 1e6,
    label="TX frequency"
)

plt.plot(
    t * 1e3,
    f_rx / 1e6,
    label="RX frequency"
)

plt.xlabel("Time (ms)")
plt.ylabel("Frequency (MHz)")

plt.title(
    "Transmitted and Delayed Received Chirps"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()