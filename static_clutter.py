"""
============================================================
PHASE 2.8 - STATIC CLUTTER
============================================================

Model:

    z[n] = z_human[n] + z_static[n] + noise[n]

The human signal contains:
    - breathing
    - heartbeat

The static signal represents:
    - wall
    - furniture
    - other stationary reflection

Goal:
    Understand how a strong stationary reflection affects
    the extracted radar phase.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Parameters
# ============================================================

slow_fs = 20
duration = 60

N = int(slow_fs * duration)

t = np.arange(N) / slow_fs


# ============================================================
# 2. Human vital-sign parameters
# ============================================================

fb = 0.25
fh = 1.20

Ab = 0.002
Ah = 0.0002


# ============================================================
# 3. Radar parameters
# ============================================================

c = 3e8
fc = 24e9

wavelength = c / fc

R0 = 3.0


# ============================================================
# 4. Human chest motion
# ============================================================

breathing = (
    Ab *
    np.cos(
        2 * np.pi *
        fb *
        t
    )
)

heartbeat = (
    Ah *
    np.cos(
        2 * np.pi *
        fh *
        t
    )
)

x = breathing + heartbeat


# ============================================================
# 5. Human range
# ============================================================

R_human = R0 + x


# ============================================================
# 6. Human phase
# ============================================================

phi_human = (
    4 * np.pi *
    R_human /
    wavelength
)


# ============================================================
# 7. Human complex signal
# ============================================================

A_human = 1.0

z_human = (
    A_human *
    np.exp(
        1j * phi_human
    )
)


# ============================================================
# 8. Static reflection
# ============================================================

# Example:
# Strong stationary reflection from a wall/furniture.

A_static = 2.0

phi_static = 1.0

z_static = (
    A_static *
    np.exp(
        1j * phi_static
    )
)

# Convert scalar into a signal with N samples

z_static_signal = (
    np.ones(N) *
    z_static
)


# ============================================================
# 9. Combined received signal
# ============================================================

z_received = (
    z_human +
    z_static_signal
)


# ============================================================
# 10. Extract phase
# ============================================================

phase_human = np.unwrap(
    np.angle(z_human)
)

phase_received = np.unwrap(
    np.angle(z_received)
)


# ============================================================
# 11. Remove mean
# ============================================================

phase_human -= np.mean(
    phase_human
)

phase_received -= np.mean(
    phase_received
)


# ============================================================
# 12. Plot human-only phase
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    t,
    phase_human,
    label="Human only"
)

plt.xlabel("Time (s)")
plt.ylabel("Phase (rad)")

plt.title(
    "Human Radar Phase Without Static Clutter"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 13. Plot phase with static clutter
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    t,
    phase_received,
    label="Human + Static Clutter"
)

plt.xlabel("Time (s)")
plt.ylabel("Phase (rad)")

plt.title(
    "Radar Phase With Static Clutter"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 14. Print information
# ============================================================

print("==========================================")
print("PHASE 2.8 - STATIC CLUTTER")
print("==========================================")

print(
    f"Human signal amplitude  : "
    f"{A_human:.2f}"
)

print(
    f"Static signal amplitude : "
    f"{A_static:.2f}"
)

print(
    f"Static/Human amplitude ratio : "
    f"{A_static/A_human:.2f}"
)

print("------------------------------------------")

print(
    "The static reflection is stationary, "
    "while the human phase varies with time."
)

print("==========================================")