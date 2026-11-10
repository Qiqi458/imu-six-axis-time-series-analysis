"""Generate reproducible synthetic six-axis IMU data."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

G = 9.80665
CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")


def generate_imu_data(sample_count: int = 1000, sample_rate_hz: float = 50.0,
                      seed: int = 42, accel_noise_std: float = 0.04,
                      gyro_noise_std: float = 0.02) -> pd.DataFrame:
    """Generate a stationary sensor in SI units; timestamp is seconds from start."""
    if sample_count < 2 or sample_rate_hz <= 0:
        raise ValueError("sample_count must be >= 2 and sample_rate_hz must be > 0")
    if accel_noise_std < 0 or gyro_noise_std < 0:
        raise ValueError("noise standard deviations cannot be negative")
    rng = np.random.default_rng(seed)
    n = np.arange(sample_count)
    return pd.DataFrame({
        "time_s": n / sample_rate_hz,
        "acc_x": rng.normal(0, accel_noise_std, sample_count),
        "acc_y": rng.normal(0, accel_noise_std, sample_count),
        "acc_z": G + rng.normal(0, accel_noise_std, sample_count),
        "gyro_x": rng.normal(0, gyro_noise_std, sample_count),
        "gyro_y": rng.normal(0, gyro_noise_std, sample_count),
        "gyro_z": rng.normal(0, gyro_noise_std, sample_count),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--rate", type=float, default=50.0, help="sample rate in Hz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/imu_data.csv"))
    args = parser.parse_args()
    df = generate_imu_data(args.samples, args.rate, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} samples to {args.output}")


if __name__ == "__main__":
    main()
