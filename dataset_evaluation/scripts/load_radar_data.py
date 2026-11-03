"""Load and inspect Position 1, 3 GHz Mendeley FMCW radar + ECG/PCG data.

Run from anywhere:
    python dataset_evaluation/scripts/load_radar_data.py

Inputs are expected under dataset_evaluation/data/ and are NOT committed to Git.
Author-provided MATLAB reference: readDCA1000.m and Data_processing.m.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]  # dataset_evaluation/
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results"

RADAR_FILE = DATA_DIR / "raw" / "adc_3GHZ_position1_ (1).bin"
TARGET1_FILE = DATA_DIR / "reference" / "log_Target1_3GHZ_position1_ (1).csv"
TARGET2_FILE = DATA_DIR / "reference" / "log_Target2_3GHZ_position1_ (1).csv"

# Dataset documentation: 60 GHz sensor, 200 ADC samples, 4 physical receivers,
# 3 transmitters (12 virtual channels), 1200 frames, 1 chirp/frame, 50 ms/frame.
FRAMES = 1200
ADC_SAMPLES = 200
PHYSICAL_RX = 4
TRANSMITTERS = 3
FRAME_PERIOD_S = 0.050
ADC_RATE_HZ = 4e6
CHIRP_SLOPE_HZ_PER_S = 60.006e12  # 3 GHz recordings, from Main.m
SPEED_OF_LIGHT = 3e8


def load_dca1000(path: Path) -> np.ndarray:
    """Return complex radar array [frame, adc_sample, selected_virtual_channel].

    Match readDCA1000.m (16-bit, complex ADC, 2I followed by 2Q), then
    Data_processing.m (keep TX1/TX3: MATLAB channels 1:4 and 9:12).
    """
    raw = np.fromfile(path, dtype="<i2")
    expected = 2 * FRAMES * ADC_SAMPLES * PHYSICAL_RX * TRANSMITTERS
    if raw.size != expected:
        raise ValueError(
            f"Unexpected BIN size: got {raw.size} int16 values; "
            f"expected {expected}. Check recording and dataset parameters."
        )

    # Every 4 int16 values represent two complex samples: I0, I1, Q0, Q1.
    iq = raw.reshape(-1, 4)
    samples = np.empty(raw.size // 2, dtype=np.complex64)
    samples[0::2] = iq[:, 0] + 1j * iq[:, 2]
    samples[1::2] = iq[:, 1] + 1j * iq[:, 3]

    # Same sample/channel/frame ordering as the supplied MATLAB code.
    virtual = samples.reshape(FRAMES, TRANSMITTERS * PHYSICAL_RX, ADC_SAMPLES)
    virtual = virtual.transpose(0, 2, 1)
    keep_channels = [0, 1, 2, 3, 8, 9, 10, 11]
    return virtual[:, :, keep_channels]


def load_reference(path: Path) -> np.ndarray:
    """CSV has two header rows: 'Column1,Column2', then 'ECG,PCG'."""
    data = np.loadtxt(path, delimiter=",", skiprows=2, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all():
        raise ValueError(f"Expected two numeric columns (ECG, PCG) in {path}")
    return data


def plot_references(ref1: np.ndarray, ref2: np.ndarray, fs_ref: float):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for row, (target_name, values) in enumerate([("Target 1", ref1), ("Target 2", ref2)]):
        n = min(len(values), int(10 * fs_ref))  # first 10 seconds
        t = np.arange(n) / fs_ref
        for column, label in enumerate(("ECG", "PCG")):
            axes[row, column].plot(t, values[:n, column], linewidth=0.9)
            axes[row, column].set_title(f"{target_name}: {label}")
            axes[row, column].set_ylabel("Recorded signal (a.u.)")
            axes[row, column].grid(alpha=0.3)
            axes[row, column].set_xlabel("Time (s)")
    fig.suptitle("Reference signals (first 10 seconds)")
    fig.tight_layout()
    dest = OUTPUT_DIR / "reference_signals.png"
    fig.savefig(dest, dpi=150)
    print(f"Saved: {dest}")


def plot_range_profile(radar: np.ndarray):
    # Diagnostic range FFT of the first of 8 selected virtual channels only.
    # This is NOT yet two-person detection or breathing-rate estimation.
    channel = radar[:, :, 0]
    channel = channel - channel.mean(axis=1, keepdims=True)  # fast-time ADC DC
    nfft = 256
    range_fft = np.fft.fft(channel, n=nfft, axis=1)[:, :nfft // 2]
    mean_magnitude = np.mean(np.abs(range_fft), axis=0)
    beat_freq_hz = np.arange(nfft // 2) * ADC_RATE_HZ / nfft
    range_m = SPEED_OF_LIGHT * beat_freq_hz / (2 * CHIRP_SLOPE_HZ_PER_S)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(range_m, 20 * np.log10(mean_magnitude + 1e-12))
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Mean FFT magnitude (dB, arbitrary reference)")
    ax.set_title("Exploratory FMCW range profile — first virtual channel")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    dest = OUTPUT_DIR / "range_profile.png"
    fig.savefig(dest, dpi=150)
    print(f"Saved: {dest}")


def main():
    for path in (RADAR_FILE, TARGET1_FILE, TARGET2_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"File missing: {path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    radar = load_dca1000(RADAR_FILE)
    ref1 = load_reference(TARGET1_FILE)
    ref2 = load_reference(TARGET2_FILE)
    radar_duration = FRAMES * FRAME_PERIOD_S
    # INFERRED, not measured: if 7500 reference samples span the 60-second radar recording.
    ref_fs = len(ref1) / radar_duration

    print("=== Vital-Radio dataset: Position 1 / 3 GHz / Test 1 ===")
    print(f"Radar I/Q shape:       {radar.shape} [frames, ADC samples, virtual channels]")
    print(f"Radar observation:     {radar_duration:.1f} seconds")
    print(f"Radar frame rate:      {1 / FRAME_PERIOD_S:.1f} Hz")
    print(f"Target 1 CSV shape:    {ref1.shape} [samples, ECG/PCG]")
    print(f"Target 2 CSV shape:    {ref2.shape} [samples, ECG/PCG]")
    print(f"Assumed reference fs:  {ref_fs:.1f} Hz (must verify against recording metadata)")
    if ref1.shape[0] != ref2.shape[0]:
        print("WARNING: The reference recordings have different lengths.")
    print("NOTE: ECG and PCG are source file labels; breathing reference is not identified yet.")

    plot_references(ref1, ref2, ref_fs)
    plot_range_profile(radar)
    plt.show()


if __name__ == "__main__":
    main()
