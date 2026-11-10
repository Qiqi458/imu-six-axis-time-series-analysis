"""Validate, filter and summarize a six-axis IMU CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")


def validate_dataset(df: pd.DataFrame) -> None:
    required = {"time_s", *CHANNELS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if len(df) < 2:
        raise ValueError("Dataset must contain at least two rows")
    values = df[["time_s", *CHANNELS]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Dataset contains missing or non-finite values")
    if np.any(np.diff(df["time_s"].to_numpy(dtype=float)) <= 0):
        raise ValueError("time_s must be strictly increasing")


def moving_average_filter(signal, win_size: int = 5) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    if not 1 <= win_size <= x.size:
        raise ValueError("win_size must be between 1 and signal length")
    # Centered rolling mean with edge-aware normalization (no artificial zero padding).
    kernel = np.ones(win_size, dtype=float)
    return np.convolve(x, kernel, mode="same") / np.convolve(np.ones_like(x), kernel, mode="same")


def analyze(input_path: Path, output_dir: Path, window: int = 5) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    validate_dataset(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = df.copy()
    for channel in CHANNELS:
        result[f"{channel}_filtered"] = moving_average_filter(df[channel], window)
    result.to_csv(output_dir / "imu_filtered.csv", index=False)
    summary = pd.DataFrame([{
        "channel": c,
        "mean": float(df[c].mean()),
        "std": float(df[c].std(ddof=0)),
        "rms": float(np.sqrt(np.mean(np.square(df[c])))),
        "filtered_std": float(result[f"{c}_filtered"].std(ddof=0)),
    } for c in CHANNELS])
    summary.to_csv(output_dir / "channel_summary.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, group, title in ((axes[0], CHANNELS[:3], "Acceleration (m/s²)"),
                               (axes[1], CHANNELS[3:], "Angular velocity (rad/s)")):
        for c in group:
            axis.plot(df["time_s"], df[c], alpha=.35, linewidth=.8, label=f"{c} raw")
            axis.plot(df["time_s"], result[f"{c}_filtered"], linewidth=1.2, label=f"{c} filtered")
        axis.set_ylabel(title)
        axis.grid(alpha=.25)
        axis.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "imu_filter_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Analyzed {len(df)} rows; outputs: {output_dir.resolve()}")
    print(summary.to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/imu_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()
    analyze(args.input, args.output_dir, args.window)


if __name__ == "__main__":
    main()
