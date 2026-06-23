"""
Baseline: integrate raw gyro to attitude (no bias removal, no filtering) and
measure how far that estimate drifts from EuRoC ground truth.

This is the anchor the learned denoiser has to beat. Raw integration keeps
every error source -- white noise, the gyro bias, and integration error -- so
the drift here is the honest "do nothing" number. On V1_01_easy that drift is
bias-dominated: the stored ground-truth gyro bias alone would rotate the frame
by hundreds of degrees over the sequence, so over the full 145 s the absolute
attitude error winds past 180 deg and a plain geodesic error saturates.

So the primary metric is per-segment, not full-track. Orientation is integrated
continuously, but error is scored on independent short windows (SEG_LEN): each
window restarts from the ground-truth attitude, integrates the gyro for SEG_LEN
seconds, and reports the geodesic error at the window end. Short windows stay
well under 180 deg, so the metric never saturates, stays comparable to the
literature, and stays sensitive to whatever the network later improves. The
full-track error is still written out for the big-picture drift curve.

Method:
  - Orientation is integrated on the quaternion manifold. For each step the
    body-frame angular velocity w[k] is held constant over dt[k], giving an
    exact incremental rotation dq = exp(0.5 * w * dt). Then q[k+1] = q[k] (x) dq
    (Hamilton product, body-frame / right multiplication).
  - A window's integration starts from the ground-truth orientation at its
    start, because a gyro measures change in orientation, not absolute
    orientation.
  - Attitude error is the geodesic angle of q_gt^-1 (x) q_est, in degrees.

Ground-truth gap: the Vicon ground truth has a ~1 s dropout, so merge_asof
pairs some IMU samples with a pose up to ~1040 ms away. Integration is never
broken (the IMU stream is continuous), but a window is skipped if either edge
falls in the dropout, so every scored number has a real reference at both ends.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_euroc_sequence

GAP_THRESH = 0.010  # s; 2x the 5 ms IMU period. Past this the paired GT pose
                    # comes from the Vicon dropout, not a real reference.
SEG_LEN = 10.0      # s; per-segment drift window. Short enough that attitude
                    # error stays well under 180 deg, so the geodesic metric
                    # never saturates and stays comparable across methods.


def quat_mult(q, r):
    """Hamilton product q (x) r, both [w, x, y, z]."""
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])


def omega_to_dquat(w, dt):
    """Exact incremental rotation for a constant body-rate w over dt."""
    theta = w * dt
    angle = np.linalg.norm(theta)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = theta / angle
    half = 0.5 * angle
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def integrate_gyro(t, gyro, q0):
    """Forward-integrate body-frame gyro into a quaternion track."""
    n = len(t)
    q = np.zeros((n, 4))
    q[0] = q0 / np.linalg.norm(q0)
    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        dq = omega_to_dquat(gyro[k], dt)
        qk = quat_mult(q[k], dq)
        q[k + 1] = qk / np.linalg.norm(qk)
    return q


def attitude_error_deg(q_est, q_gt):
    """Geodesic angle between two quaternion tracks, per sample, in degrees."""
    err = np.empty(len(q_est))
    for k in range(len(q_est)):
        g = q_gt[k]
        ginv = np.array([g[0], -g[1], -g[2], -g[3]])  # unit quat inverse
        qe = quat_mult(ginv, q_est[k])
        w = min(1.0, abs(qe[0]))
        err[k] = np.degrees(2.0 * np.arccos(w))
    return err


def segment_errors(t, gyro, q_gt, gap, seg_len):
    """Independent short-window drift. Each window restarts from GT attitude,
    integrates gyro for seg_len seconds, and reports the geodesic error at the
    window end. A window is skipped if either edge falls in the GT dropout, so
    every reported number has a real reference at both ends."""
    errs, starts = [], []
    n = len(t)
    i = 0
    while i < n - 1:
        t_end = t[i] + seg_len
        j = np.searchsorted(t, t_end)
        if j >= n:
            break
        if gap[i] <= GAP_THRESH and gap[j] <= GAP_THRESH:
            q_seg = integrate_gyro(t[i:j + 1], gyro[i:j + 1], q_gt[i])
            e = attitude_error_deg(q_seg[-1:], q_gt[j:j + 1])[0]
            errs.append(e)
            starts.append(t[i])
        i = j  # non-overlapping windows
    return np.array(starts), np.array(errs)


def main(seq_dir, outdir="results"):
    m = load_euroc_sequence(seq_dir)
    t = m["t"].values
    gyro = m[["wx", "wy", "wz"]].values
    q_gt = m[["qw", "qx", "qy", "qz"]].values
    gap = np.abs(m["t"].values - m["t_gt"].values)

    # Full-track integration: drift curve, can wind past 180 deg.
    q_est = integrate_gyro(t, gyro, q_gt[0])
    err_full = attitude_error_deg(q_est, q_gt)

    # Primary metric: independent short-window drift (never saturates).
    starts, seg_err = segment_errors(t, gyro, q_gt, gap, SEG_LEN)

    print(f"samples              : {len(m)}")
    print(f"segment length       : {SEG_LEN:.0f} s")
    print(f"segments scored      : {len(seg_err)}  "
          f"(skipped where GT dropout hits a window edge)")
    print(f"per-{SEG_LEN:.0f}s drift mean   : {seg_err.mean():.2f} deg")
    print(f"per-{SEG_LEN:.0f}s drift median : {np.median(seg_err):.2f} deg")
    print(f"per-{SEG_LEN:.0f}s drift max    : {seg_err.max():.2f} deg")
    print(f"per-{SEG_LEN:.0f}s drift min    : {seg_err.min():.2f} deg")

    out = Path(outdir)
    out.mkdir(exist_ok=True)
    csv = out / "baseline_segment_drift.csv"
    np.savetxt(csv, np.column_stack([starts, seg_err]), delimiter=",",
               header="seg_start_s,drift_deg", comments="", fmt="%.6f")
    print(f"wrote {csv}")

    csv2 = out / "baseline_attitude_error.csv"
    np.savetxt(csv2, np.column_stack([t, err_full, (gap <= GAP_THRESH).astype(int)]),
               delimiter=",", header="t,err_deg,valid", comments="",
               fmt=["%.6f", "%.6f", "%d"])
    print(f"wrote {csv2}  (full-track, can exceed 180 deg via wraparound)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python src/integrate_baseline.py <sequence_dir>")
        sys.exit(1)
    main(sys.argv[1])