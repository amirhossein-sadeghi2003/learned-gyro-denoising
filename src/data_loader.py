"""
Load an EuRoC MAV sequence (ASL format): IMU + ground-truth pose.

Layout:
  <seq>/mav0/imu0/data.csv
  <seq>/mav0/state_groundtruth_estimate0/data.csv

IMU runs at 200 Hz; ground truth comes from a separate clock, so each IMU
sample is paired with the nearest-in-time ground-truth pose. Timestamps are
in nanoseconds and converted to seconds relative to the IMU start.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

IMU_COLS = ["t_ns", "wx", "wy", "wz", "ax", "ay", "az"]
GT_COLS = ["t_ns", "px", "py", "pz", "qw", "qx", "qy", "qz",
           "vx", "vy", "vz", "bwx", "bwy", "bwz", "bax", "bay", "baz"]


def _read_euroc_csv(path, names):
    return pd.read_csv(path, comment="#", header=None, names=names)


def load_euroc_sequence(seq_dir):
    seq = Path(seq_dir)
    imu = _read_euroc_csv(seq / "mav0" / "imu0" / "data.csv", IMU_COLS)
    gt = _read_euroc_csv(
        seq / "mav0" / "state_groundtruth_estimate0" / "data.csv", GT_COLS)

    t0 = imu["t_ns"].iloc[0]
    for df in (imu, gt):
        df.sort_values("t_ns", inplace=True)
        df["t"] = (df["t_ns"] - t0) * 1e-9

    merged = pd.merge_asof(imu, gt, on="t_ns", direction="nearest",
                           suffixes=("", "_gt"))
    return merged


def _main(seq_dir):
    seq = Path(seq_dir)
    imu_path = seq / "mav0" / "imu0" / "data.csv"
    gt_path = seq / "mav0" / "state_groundtruth_estimate0" / "data.csv"

    print("IMU header:", imu_path.read_text().splitlines()[0])
    print("GT  header:", gt_path.read_text().splitlines()[0])
    print()

    m = load_euroc_sequence(seq_dir)
    dt = np.diff(m["t"].values)
    print(f"samples: {len(m)}")
    print(f"duration: {m['t'].iloc[-1]:.1f} s")
    print(f"IMU rate: {1.0 / np.median(dt):.1f} Hz "
          f"(median dt = {np.median(dt) * 1e3:.3f} ms)")

    wmag = np.linalg.norm(m[["wx", "wy", "wz"]].values, axis=1)
    qnorm = np.linalg.norm(m[["qw", "qx", "qy", "qz"]].values, axis=1)
    print(f"gyro |w|  : mean {wmag.mean():.3f}, max {wmag.max():.3f} rad/s")
    print(f"quat norm : mean {qnorm.mean():.4f} (should be ~1.0)")

    gap = np.abs(m["t"].values - m["t_gt"].values)
    print(f"IMU<->GT time gap: median {np.median(gap) * 1e3:.2f} ms, "
          f"max {gap.max() * 1e3:.2f} ms")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python src/data_loader.py <sequence_dir>")
        sys.exit(1)
    _main(sys.argv[1])
