"""Run the complete synthetic IMU demo in pipeline order."""
from __future__ import annotations

import argparse
from pathlib import Path

from imu_data_generate import generate_imu_data
from main_sensor_study import analyze as analyze_basic
from imu_advanced_signal_analysis import analyze as analyze_advanced
from imu_multi_condition_analysis import run as analyze_conditions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    source = root / "imu_data.csv"
    generate_imu_data(args.samples, args.rate, args.seed).to_csv(source, index=False)
    analyze_basic(source, root / "analysis")
    analyze_advanced(source, root / "advanced")
    analyze_conditions(root / "conditions", args.samples, args.rate, args.seed)
    print(f"Complete pipeline finished. Results: {root.resolve()}")


if __name__ == "__main__":
    main()
