"""
Phase 2.1 - FMCW Transmitter Simulation

Purpose:
    Generate and visualize an FMCW chirp based on the
    mathematical model developed in Phase 1.

Mathematical model:

    s_tx(t) = A * exp(j*2*pi*(f0*t + (S/2)*t^2))

    S = B / Tc

    f(t) = f0 + S*t

Important:
    The parameter values used here are illustrative simulation
    values only. They are NOT the final EV-TINYRAD24G hardware
    parameters.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Define simulation parameters
# ============================================================

# Signal amplitude
A = 1.0

# Starting frequency
# This is an illustrative simulation value.
f0 = 1e6                  # 1 MHz

# FMCW sweep bandwidth
B = 2e6                   # 2 MHz

# Chirp duration
Tc = 1e-3                # 1 ms

# Chirp slope
S = B / Tc

# Sampling frequency
# Must be sufficiently larger than the simulated frequency
fs = 10e6                # 10 MHz

# Number of samples
N = int(Tc * fs)


# ============================================================
# 2. Create the time vector
# ============================================================

t = np.arange(N) / fs


# ============================================================
# 3. Calculate instantaneous frequency
# ============================================================

f_inst = f0 + S * t


# ============================================================
# 4. Generate the complex FMCW chirp
# ============================================================

phase_tx = 2 * np.pi * (
    f0 * t + (S / 2) * t**2
)

s_tx = A * np.exp(1j * phase_tx)


# ============================================================
# 5. Separate the I and Q components
# ============================================================

I_tx = np.real(s_tx)
Q_tx = np.imag(s_tx)


# ============================================================
# 6. Print important parameters
# ============================================================

print("==========================================")
print("FMCW TRANSMITTER SIMULATION")
print("==========================================")

print(f"Starting frequency : {f0/1e6:.2f} MHz")
print(f"Bandwidth          : {B/1e6:.2f} MHz")
print(f"Chirp duration     : {Tc*1e3:.2f} ms")
print(f"Chirp slope        : {S/1e9:.2f} GHz/s")
print(f"Sampling frequency : {fs/1e6:.2f} MHz")
print(f"Number of samples  : {N}")

print("------------------------------------------")
print(f"Initial frequency  : {f_inst[0]/1e6:.2f} MHz")
print(f"Final frequency    : {f_inst[-1]/1e6:.2f} MHz")
print("==========================================")


# ============================================================
# 7. Plot instantaneous frequency
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(t * 1e3, f_inst / 1e6)

plt.xlabel("Time (ms)")
plt.ylabel("Instantaneous Frequency (MHz)")
plt.title("FMCW Chirp - Instantaneous Frequency")

plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 8. Plot the real part of the transmitted signal
# ============================================================

# Show only the first 10 microseconds
zoom_time = 10e-6

zoom_samples = t <= zoom_time

plt.figure(figsize=(10, 5))

plt.plot(
    t[zoom_samples] * 1e6,
    I_tx[zoom_samples]
)

plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.title("FMCW Transmitted Signal - Zoomed")

plt.grid(True)
plt.tight_layout()
plt.show()