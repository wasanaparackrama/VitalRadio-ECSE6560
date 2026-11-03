"""
Phase 2.4 - FMCW Range FFT

Purpose:
    Estimate the target range from the beat signal.

Processing:

    Beat signal
        ↓
       FFT
        ↓
    Beat frequency
        ↓
    Range equation

    R = c * f_beat / (2*S)
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Constants
# ============================================================

c = 3e8


# ============================================================
# 2. FMCW parameters
# ============================================================

A = 1.0

f0 = 1e6
B = 100e6
Tc = 1e-3

S = B / Tc


# ============================================================
# 3. Target
# ============================================================

R0 = 3.0

alpha = 0.8


# ============================================================
# 4. Sampling
# ============================================================

fs = 250e6

N = int(Tc * fs)

t = np.arange(N) / fs


# ============================================================
# 5. Calculate propagation delay
# ============================================================

tau = (2 * R0) / c

delay_samples = int(round(tau * fs))


# ============================================================
# 6. Generate transmitted FMCW signal
# ============================================================

phase_tx = 2 * np.pi * (
    f0 * t + (S / 2) * t**2
)

s_tx = A * np.exp(1j * phase_tx)


# ============================================================
# 7. Generate received signal
# ============================================================

s_rx = np.zeros_like(s_tx)

s_rx[delay_samples:] = (
    alpha * s_tx[:-delay_samples]
)


# ============================================================
# 8. Generate beat signal
# ============================================================

s_beat = np.zeros_like(s_tx)

s_beat[delay_samples:] = (
    s_tx[delay_samples:]
    * np.conj(s_rx[delay_samples:])
)


# ============================================================
# 9. FFT of beat signal
# ============================================================

# FFT
fft_result = np.fft.fft(s_beat)

# Corresponding frequency axis
freq = np.fft.fftfreq(N, d=1/fs)


# ============================================================
# 10. Calculate magnitude spectrum
# ============================================================

fft_magnitude = np.abs(fft_result) / N


# ============================================================
# 11. Use only positive frequencies
# ============================================================

positive = freq >= 0

freq_positive = freq[positive]
magnitude_positive = fft_magnitude[positive]


# ============================================================
# 12. Find the strongest frequency peak
# ============================================================

peak_index = np.argmax(magnitude_positive)

f_beat_estimated = freq_positive[peak_index]


# ============================================================
# 13. Calculate estimated range
# ============================================================

R_estimated = (
    c * f_beat_estimated
    / (2 * S)
)


# ============================================================
# 14. Theoretical values
# ============================================================

f_beat_theoretical = S * tau


# ============================================================
# 15. Calculate range error
# ============================================================

range_error = R_estimated - R0

range_error_percent = (
    abs(range_error)
    / R0
    * 100
)


# ============================================================
# 16. Print results
# ============================================================

print("==========================================")
print("PHASE 2.4 - RANGE FFT")
print("==========================================")

print(f"Actual target range       : {R0:.4f} m")

print("------------------------------------------")

print(
    f"Theoretical beat frequency: "
    f"{f_beat_theoretical/1e3:.4f} kHz"
)

print(
    f"Estimated beat frequency  : "
    f"{f_beat_estimated/1e3:.4f} kHz"
)

print("------------------------------------------")

print(
    f"Estimated target range    : "
    f"{R_estimated:.4f} m"
)

print(
    f"Range error               : "
    f"{range_error:.6f} m"
)

print(
    f"Range error percentage    : "
    f"{range_error_percent:.4f} %"
)

print("==========================================")


# ============================================================
# 17. Plot the FFT spectrum
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    freq_positive / 1e3,
    magnitude_positive
)

plt.axvline(
    f_beat_theoretical / 1e3,
    linestyle="--",
    label="Theoretical beat frequency"
)

plt.axvline(
    f_beat_estimated / 1e3,
    linestyle=":",
    label="Estimated beat frequency"
)

plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude")

plt.title("Beat Signal FFT - Range Estimation")

plt.xlim(0, 10)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()