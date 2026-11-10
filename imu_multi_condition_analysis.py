"""Compare stationary, vibration and pulse-shock synthetic IMU conditions."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

G = 9.80665


def generate_imu(sample_count: int, sample_rate_hz: float, noise_std: float,
                 name: str, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if sample_count < 2 or sample_rate_hz <= 0 or noise_std < 0:
        raise ValueError("Require samples >= 2, positive rate and non-negative noise")
    rng = np.random.default_rng(seed)
    t = np.arange(sample_count) / sample_rate_hz
    truth = np.zeros(sample_count, dtype=bool)
    ax = rng.normal(0, noise_std, sample_count)
    if name == "light_vibration":
        ax += 0.12 * np.sin(2 * np.pi * 4.0 * t)
    elif name == "shock":
        # Two deterministic half-sine pulses provide event ground truth.
        for center in (sample_count // 3, 2 * sample_count // 3):
            width = max(3, int(round(0.08 * sample_rate_hz)))
            left, right = max(0, center - width // 2), min(sample_count, center + width // 2 + 1)
            phase = np.linspace(0, np.pi, right - left)
            ax[left:right] += 1.2 * np.sin(phase)
            truth[left:right] = True
    df = pd.DataFrame({
        "time_s": t, "acc_x": ax,
        "acc_y": rng.normal(0, noise_std, sample_count),
        "acc_z": G + rng.normal(0, noise_std, sample_count),
        "gyro_x": rng.normal(0, noise_std * .8, sample_count),
        "gyro_y": rng.normal(0, noise_std * .8, sample_count),
        "gyro_z": rng.normal(0, noise_std * .8, sample_count),
    })
    return df, truth


def moving_average(signal, win_size: int = 7) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or not x.size or not 1 <= win_size <= x.size:
        raise ValueError("Expected non-empty 1D signal and valid window")
    return np.convolve(x, np.ones(win_size), mode="same") / np.convolve(np.ones_like(x), np.ones(win_size), mode="same")


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(truth & predicted)); fp = int(np.sum(~truth & predicted)); fn = int(np.sum(truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def run(output_dir: Path, samples: int = 500, rate: float = 50.0, seed: int = 42,
        threshold: float = 0.35) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = (("static", .03), ("light_vibration", .04), ("shock", .04))
    reports = []
    for offset, (name, noise) in enumerate(conditions):
        df, truth = generate_imu(samples, rate, noise, name, seed + offset)
        filt = moving_average(df["acc_x"], 7)
        predicted = np.abs(filt) >= threshold
        metrics = binary_metrics(truth, predicted) if name == "shock" else {"precision": np.nan, "recall": np.nan, "f1": np.nan, "tp": 0, "fp": int(predicted.sum()), "fn": 0}
        df["acc_x_filtered"] = filt
        df["shock_truth"] = truth
        df["shock_detected"] = predicted
        df.to_csv(output_dir / f"imu_{name}.csv", index=False)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(df["time_s"], df["acc_x"], alpha=.45, label="raw acc_x")
        ax.plot(df["time_s"], filt, label="7-sample moving average")
        ax.axhline(threshold, color="r", linestyle="--", linewidth=.8)
        ax.axhline(-threshold, color="r", linestyle="--", linewidth=.8)
        ax.fill_between(df["time_s"], -1.5, 1.5, where=truth, alpha=.12, color="orange", label="shock ground truth")
        ax.set_title(f"{name}: synthetic IMU condition"); ax.set_xlabel("Time (s)"); ax.grid(alpha=.25); ax.legend()
        fig.tight_layout(); fig.savefig(output_dir / f"plot_{name}.png", dpi=180); plt.close(fig)
        reports.append({"condition": name, "acc_x_std": float(df["acc_x"].std(ddof=0)),
                        "z_bias_estimate": float(df["acc_z"].mean() - G),
                        "detected_samples": int(predicted.sum()), **metrics})
    report = pd.DataFrame(reports)
    report.to_csv(output_dir / "condition_report.csv", index=False)
    print(report.to_string(index=False))
    print(f"Outputs: {output_dir.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/conditions"))
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=.35)
    args = parser.parse_args()
    run(args.output_dir, args.samples, args.rate, args.seed, args.threshold)


if __name__ == "__main__":
    main()
