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

f0 = 1e6                 # 1 MHz
B = 100e6                # 100 MHz
Tc = 1e-3                # 1 ms

S = B / Tc


# ============================================================
# 3. Target
# ============================================================

R0 = 3.0                 # 3 meters
alpha = 0.8


# ============================================================
# 4. Sampling
# ============================================================

fs = 250e6

N = int(Tc * fs)

t = np.arange(N) / fs


# ============================================================
# 5. Propagation delay
# ============================================================

tau = (2 * R0) / c

delay_samples = int(round(tau * fs))


# ============================================================
# 6. Generate transmitted chirp
# ============================================================

phase_tx = 2 * np.pi * (
    f0 * t + (S / 2) * t**2
)

s_tx = A * np.exp(1j * phase_tx)


# ============================================================
# 7. Generate delayed received chirp
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
# 9. Theoretical beat frequency
# ============================================================

f_beat_theoretical = S * tau


# ============================================================
# 10. Print information
# ============================================================

print("==========================================")
print("PHASE 2.3 - BEAT SIGNAL")
print("==========================================")

print(f"Target range       : {R0:.2f} m")
print(f"Round-trip delay   : {tau*1e9:.2f} ns")
print(f"Delay samples      : {delay_samples}")
print(f"Chirp slope        : {S/1e9:.2f} GHz/s")

print("------------------------------------------")

print(
    f"Theoretical beat frequency : "
    f"{f_beat_theoretical/1e3:.2f} kHz"
)

print("==========================================")


# ============================================================
# 11. Plot real part of beat signal
# ============================================================

# Only plot a short section so we can see the waveform
zoom_time = 5e-3

zoom = t <= zoom_time

plt.figure(figsize=(10, 5))

plt.plot(
    t[zoom] * 1e3,
    np.real(s_beat[zoom])
)

plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")

plt.title("FMCW Beat Signal")

plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 12. Plot beat signal magnitude
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    t[zoom] * 1e3,
    np.abs(s_beat[zoom])
)

plt.xlabel("Time (ms)")
plt.ylabel("Magnitude")

plt.title("Beat Signal Magnitude")

plt.grid(True)

plt.tight_layout()
plt.show()