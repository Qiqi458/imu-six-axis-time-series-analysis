"""Time-aware IMU diagnostics: calibration estimate, integration and event grouping."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

G = 9.80665
ACC = ("acc_x", "acc_y", "acc_z")
GYRO = ("gyro_x", "gyro_y", "gyro_z")


def validate(df: pd.DataFrame) -> None:
    required = {"time_s", *ACC, *GYRO}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    v = df[list(required)].to_numpy(dtype=float)
    if len(df) < 2 or not np.isfinite(v).all():
        raise ValueError("Need at least two rows of finite data")
    if np.any(np.diff(df["time_s"]) <= 0):
        raise ValueError("time_s must be strictly increasing")


def detect_events(df: pd.DataFrame, channel: str = "gyro_x", threshold: float = 0.15,
                  min_duration_s: float = 0.0, merge_gap_s: float = 0.04) -> pd.DataFrame:
    """Group adjacent threshold crossings into events; durations are in seconds."""
    if channel not in df or threshold <= 0 or min_duration_s < 0 or merge_gap_s < 0:
        raise ValueError("Invalid event channel or threshold/duration parameters")
    t = df["time_s"].to_numpy(float)
    mask = np.abs(df[channel].to_numpy(float)) >= threshold
    idx = np.flatnonzero(mask)
    if not idx.size:
        return pd.DataFrame(columns=["event_id", "channel", "start_s", "end_s", "duration_s", "peak_abs", "peak_time_s", "sample_count"])
    nominal_dt = float(np.median(np.diff(t)))
    # A missing run is merged only when the elapsed time between thresholded
    # samples is no greater than one normal sample interval plus the allowed gap.
    groups = np.split(idx, np.flatnonzero(np.diff(t[idx]) > nominal_dt + merge_gap_s) + 1)
    rows = []
    for group in groups:
        start, end = t[group[0]], t[group[-1]]
        duration = end - start
        if duration < min_duration_s:
            continue
        peak_idx = group[np.argmax(np.abs(df[channel].to_numpy(float)[group]))]
        rows.append({"event_id": len(rows) + 1, "channel": channel, "start_s": start,
                     "end_s": end, "duration_s": duration,
                     "peak_abs": abs(df[channel].iloc[peak_idx]), "peak_time_s": t[peak_idx],
                     "sample_count": len(group)})
    return pd.DataFrame(rows, columns=["event_id", "channel", "start_s", "end_s", "duration_s", "peak_abs", "peak_time_s", "sample_count"])


def analyze(input_path: Path, output_dir: Path, threshold: float = 0.15) -> None:
    df = pd.read_csv(input_path)
    validate(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    t = df["time_s"].to_numpy(float)
    dt = np.diff(t)
    # Trapezoidal integration starts at zero. No gravity removal or attitude compensation is implied.
    result = df.copy()
    result["sample_interval_s"] = np.r_[np.nan, dt]
    for axis in ACC:
        result[f"{axis}_delta_per_s"] = np.gradient(df[axis].to_numpy(float), t)
        result[f"{axis}_velocity_raw"] = np.concatenate(([0.0], np.cumsum((df[axis].to_numpy(float)[1:] + df[axis].to_numpy(float)[:-1]) * dt / 2)))
    # Estimate stationary offsets from the initial window; for acc_z this removes gravity as well.
    n_static = max(2, min(len(df) // 10, int(round(2.0 / np.median(dt)))))
    baseline = {axis: float(df[axis].iloc[:n_static].mean()) for axis in ACC}
    for axis in ACC:
        result[f"{axis}_baseline_removed"] = df[axis] - baseline[axis]
        corrected = result[f"{axis}_baseline_removed"].to_numpy(float)
        result[f"{axis}_velocity_corrected"] = np.concatenate(([0.0], np.cumsum((corrected[1:] + corrected[:-1]) * dt / 2)))
    events = detect_events(df, threshold=threshold)
    result.to_csv(output_dir / "imu_advanced_result.csv", index=False)
    events.to_csv(output_dir / "imu_events.csv", index=False)
    pd.DataFrame([{"channel": k, "initial_window_mean": v, "window_samples": n_static} for k, v in baseline.items()]).to_csv(output_dir / "initial_baseline_estimate.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t, df["acc_z"], label="acc_z raw")
    axes[0].plot(t, result["acc_z_baseline_removed"], label="initial-mean removed")
    axes[0].legend(); axes[0].grid(alpha=.25)
    for axis in ACC:
        axes[1].plot(t, result[f"{axis}_velocity_raw"], label=axis)
    axes[1].set_title("Raw acceleration integral (drift expected)"); axes[1].legend(); axes[1].grid(alpha=.25)
    axes[2].plot(t, df["gyro_x"], label="gyro_x")
    axes[2].axhline(threshold, color="r", linestyle="--"); axes[2].axhline(-threshold, color="r", linestyle="--")
    axes[2].set_xlabel("Time (s)"); axes[2].legend(); axes[2].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output_dir / "advanced_diagnostics.png", dpi=180); plt.close(fig)
    print(f"Analyzed {len(df)} samples; {len(events)} grouped gyro_x events; outputs: {output_dir.resolve()}")
    print("Initial means are a simple stationary-window estimate, not a full IMU calibration.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/imu_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/advanced"))
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()
    analyze(args.input, args.output_dir, args.threshold)


if __name__ == "__main__":
    main()
