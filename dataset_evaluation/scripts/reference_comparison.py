"""Exploratory comparison of radar spectral peaks with ECG reference heart rates.

Run from the repository root:
    python dataset_evaluation/scripts/reference_comparison.py

Prerequisites:
    python dataset_evaluation/scripts/vital_sign_estimation.py
    python dataset_evaluation/scripts/reference_validation.py

Position 1 geometry (dataset documentation): Target1 ~40 deg, Target2 ~20 deg.
This comparison matches by ABSOLUTE angle only. It does NOT validate the
array's sign convention, person identity, or radar heartbeat detection.
It compares both preprocessing methods without choosing one per person.
"""

from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # dataset_evaluation/
RESULTS = ROOT / "results"
RADAR_CSV = RESULTS / "vital_sign_diagnostic_estimates.csv"
ECG_CSV = RESULTS / "reference_ecg_summary.csv"
TARGET_ANGLE_ABS_DEG = {1: 40.0, 2: 20.0}  # Position 1 diagram
PREPROCESSING = ("Original", "Mean cancelled")


def load_ecg():
    with ECG_CSV.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        label = row.get("reference_target", row.get("Target", ""))
        target = int("".join(c for c in label if c.isdigit()))
        key = "beat_count_per_min" if "beat_count_per_min" in row else "Beat Count BPM"
        out[target] = float(row[key])
    if set(out) != {1, 2}:
        raise ValueError("Expected exactly Target1 and Target2 in reference ECG summary")
    return out


def load_radar():
    with RADAR_CSV.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out, angles = {}, {}
    for row in rows:
        if row["frequency_band"] != "Heartbeat":
            continue
        candidate = int(row["candidate"])
        angles[candidate] = float(row["estimated_angle_deg"])
        out[(candidate, row["preprocessing"])] = float(row["spectral_peak_per_min"])
    if set(angles) != {1, 2}:
        raise ValueError("Expected exactly 2 radar candidates")
    for candidate in angles:
        for method in PREPROCESSING:
            if (candidate, method) not in out:
                raise ValueError(f"Missing {method} measurement for candidate {candidate}")
    return out, angles


def angle_based_hypothesis(angles):
    """Match two candidates using closeness of |azimuth| to nominal geometry."""
    candidates = sorted(angles)
    hypotheses = [
        {1: candidates[0], 2: candidates[1]},
        {1: candidates[1], 2: candidates[0]},
    ]
    return min(hypotheses, key=lambda h: sum(
        abs(abs(angles[h[t]]) - TARGET_ANGLE_ABS_DEG[t]) for t in (1, 2)
    ))


def main():
    if not ECG_CSV.exists() or not RADAR_CSV.exists():
        raise FileNotFoundError("Run reference_validation.py and vital_sign_estimation.py first.")
    ecg = load_ecg()
    radar, angles = load_radar()
    assignment = angle_based_hypothesis(angles)
    rows = []

    print("=== EXPLORATORY heartbeat comparison: NOT validated radar accuracy ===")
    print("Reference ECG sampling: 125 Hz (8 ms) documented by Lei et al. (2024).")
    print("Candidate assignment uses POSITION 1 ANGLE MAGNITUDES only, NOT heart rates.")
    for t in (1, 2):
        c = assignment[t]
        print(f"Target{t} (~{TARGET_ANGLE_ABS_DEG[t]:.0f} deg) <- "
              f"Candidate{c} ({angles[c]:+.0f} deg): HYPOTHESIS ONLY")
        for method in PREPROCESSING:
            r = radar[c, method]
            error = abs(r - ecg[t])
            print(f"  {method:15s}: radar band peak {r:5.1f}/min, "
                  f"ECG {ecg[t]:5.1f} BPM, absolute difference {error:5.1f}")
            rows.append({
                "reference_target": f"Target{t}",
                "hypothesized_candidate": f"Candidate{c}",
                "candidate_angle_deg": angles[c],
                "preprocessing": method,
                "ecg_reference_bpm": ecg[t],
                "radar_heartbeat_band_peak_per_min": r,
                "exploratory_absolute_difference_per_min": error,
                "status": "UNVALIDATED subject assignment and radar heartbeat peak",
            })
    for method in PREPROCESSING:
        mae = np.mean([x["exploratory_absolute_difference_per_min"]
                       for x in rows if x["preprocessing"] == method])
        print(f"Pilot mean absolute DIFFERENCE for {method}: {mae:.1f}/min "
              "(two subjects, ONE recording; not a validated accuracy metric)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = RESULTS / "reference_comparison_exploratory.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
    for ax, t in zip(axes, (1, 2)):
        c = assignment[t]
        x = np.arange(3)
        vals = [ecg[t], radar[c, "Original"], radar[c, "Mean cancelled"]]
        bars = ax.bar(x, vals)
        ax.set_xticks(x, ["ECG\nreference", "Radar\noriginal", "Radar\nmean cancelled"])
        ax.set_title(f"Target{t} vs Candidate{c} ({angles[c]:+.0f} deg)\n"
                     "Assignment by nominal angle only")
        ax.set_ylim(0, max(ecg.values()) * 1.25)
        ax.set_ylabel("Heart rate / heartbeat-band peak (per minute)")
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.0f}",
                    ha="center", va="bottom")
    fig.suptitle("Exploratory ECG–radar comparison: NOT validated heartbeat accuracy")
    figure_file = RESULTS / "reference_comparison_exploratory.png"
    fig.savefig(figure_file, dpi=160)
    plt.close(fig)
    print(f"Saved: {summary}\nSaved: {figure_file}")
    print("Do not select different methods per subject to optimize agreement.")
    print("Verify candidate identity and repeat across recordings before reporting MAE/accuracy.")


if __name__ == "__main__":
    main()
