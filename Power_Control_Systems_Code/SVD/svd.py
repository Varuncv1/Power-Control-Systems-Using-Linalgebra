"""
replicate_sanandaji_svd_acd_paper.py
------------------------------------
Paper-faithful abrupt change detection via SVD of the *difference* history matrix.

Δ_t = [ (y_t - y_{t-1}), (y_t - y_{t-2}), ..., (y_t - y_{t-w}) ]  ∈ R^{M×w}
We track σ1(Δ_t) and raise detections when it exceeds a robust baseline threshold.

Usage (synthetic demo):
    python replicate_sanandaji_svd_acd_paper.py --demo --win 16 --baseline 200 --k 6 --method mad

Usage (CSV):
    python replicate_sanandaji_svd_acd_paper.py --csv your.csv --time-col time --channels V1 V2 I1 I2 \
        --win 16 --baseline 200 --k 6 --method mad --norm z --save-prefix out

Reference: Sanandaji et al., “An Abrupt Change Detection Heuristic…” (uploaded paper)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- utils ----------

def sanitize_dataframe(df, channels):
    df = df.copy()
    for c in channels:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[channels] = df[channels].ffill().bfill()
    return df

def zscore_baseline(X, baseline_len):
    """Per-channel z-score using only the first 'baseline_len' samples."""
    X0 = X[:baseline_len]
    mu = np.nanmean(X0, axis=0)
    sd = np.nanstd(X0, axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd

# ---------- paper-faithful Δ_t and σ1(Δ_t) ----------

def sigma1_history_matrix_from_diffs(X, win):
    """
    Δ_t[:, k-1] = y_t - y_{t-k},  k=1..win
    X: [T, M] (rows=time, cols=channels)
    Returns s1[t] = σ1(Δ_t); np.nan for t < win
    """
    T, M = X.shape
    s1 = np.full(T, np.nan)
    for t in range(win, T):
        Dt = np.empty((M, win), dtype=float)
        y_t = X[t]  # (M,)
        for k in range(1, win + 1):
            Dt[:, k - 1] = y_t - X[t - k]
        try:
            # we only need the largest singular value
            _, S, _ = np.linalg.svd(Dt, full_matrices=False)
            s1[t] = S[0]
        except np.linalg.LinAlgError:
            s1[t] = np.nan
    return s1

# ---------- thresholding ----------

def robust_threshold(series, baseline_len, k=6.0, method="mad"):
    base = series[:baseline_len]
    base = base[np.isfinite(base)]
    if base.size == 0:
        return np.nan
    if method.lower() == "std":
        mu = np.nanmean(base); sd = np.nanstd(base, ddof=1)
        return mu + k * sd
    med = np.nanmedian(base); mad = np.nanmedian(np.abs(base - med))
    scale = 1.4826 * mad if mad > 0 else np.nanstd(base, ddof=1)
    return med + k * scale

def detect_spikes(s1, thr):
    return np.isfinite(s1) & (s1 > thr)

# ---------- demo data ----------

def make_synthetic_demo(T=800, M=20, change_at=400, seed=1):
    """
    Simple linear measurement model with a step attack at t=change_at on first 4 channels.
    Produces a rank-1-like effect that spikes σ1(Δ_t) then fades by ~ t_a + win.
    """
    rng = np.random.default_rng(seed)
    # slow-varying latent state
    x = np.zeros((T, 5))
    for t in range(1, T):
        x[t] = x[t-1] + 0.01 * rng.normal(size=5)
    H = rng.normal(scale=0.5, size=(M, x.shape[1]))
    y = (H @ x.T).T
    y += 0.02 * rng.normal(size=y.shape)
    a = np.zeros(M); a[:4] = 0.8  # sparse step attack
    y[change_at:] += a
    df = pd.DataFrame(y, columns=[f"m{j+1}" for j in range(M)])
    df["t"] = np.arange(T)
    return df

# ---------- pipeline ----------

def run_pipeline(df, time_col, channels, win=16, baseline_len=200, k=6.0,
                 norm="z", thresh_method="mad", title_suffix="(paper-faithful)"):
    df = sanitize_dataframe(df, channels)
    X = df[channels].values.astype(float)  # T x M

    # optional per-channel z-score from baseline
    if norm == "z":
        Xz, _, _ = zscore_baseline(X, baseline_len)
    else:
        Xz = X

    s1 = sigma1_history_matrix_from_diffs(Xz, win)
    thr = robust_threshold(s1, baseline_len, k=k, method=thresh_method)
    det = detect_spikes(s1, thr)

    # time axis
    t = pd.to_datetime(df[time_col]) if (time_col and time_col in df) else np.arange(len(df))

    # ---- summary prints (always) ----
    n_det = int(np.nansum(det))
    first_idx = int(np.where(det)[0][0]) if n_det > 0 else None
    first_time = (t[first_idx] if first_idx is not None else None)
    print(f"[ACD] win={win}, baseline={baseline_len}, k={k}, method={thresh_method}, norm={norm}")
    print(f"[ACD] threshold={thr:.4f}  detections={n_det}  first_idx={first_idx}  first_time={first_time}")

    # ---- plot (always) ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, s1, lw=1.6, label=r"$\sigma_1(\Delta_t)$")
    if np.isfinite(thr):
        ax.axhline(thr, ls="--", label="threshold")
    if n_det > 0:
        ax.plot(t[det], s1[det], "o", ms=3, label="detections")
    ax.set_title(f"SVD ACD on Δ_t {title_suffix}  |  win={win}, baseline={baseline_len}, k={k}, {thresh_method.upper()}")
    ax.set_xlabel("time"); ax.set_ylabel("largest singular value")
    ax.legend(); fig.tight_layout()

    return {"s1": s1, "threshold": thr, "detections": det, "time": t, "figure": fig}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run synthetic demo")
    ap.add_argument("--csv", type=str, default=None, help="path to PMU/SCADA CSV")
    ap.add_argument("--time-col", type=str, default=None, help="timestamp column name")
    ap.add_argument("--channels", nargs="+", default=None, help="channel column names")
    ap.add_argument("--win", type=int, default=16, help="history window length w")
    ap.add_argument("--baseline", type=int, default=200, help="baseline samples")
    ap.add_argument("--k", type=float, default=6.0, help="threshold multiplier")
    ap.add_argument("--method", type=str, default="mad", choices=["mad","std"], help="threshold method")
    ap.add_argument("--norm", type=str, default="z", choices=["z","none"], help="per-channel baseline z-score")
    ap.add_argument("--save-prefix", type=str, default=None, help="prefix to save PNG (optional)")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.demo:
        df = make_synthetic_demo()
        time_col = "t"
        channels = [c for c in df.columns if c != time_col]
    elif args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)
        time_col = args.time_col if args.time_col else None
        channels = args.channels if args.channels else df.select_dtypes(include=["number"]).columns.drop(time_col).tolist()
    else:
        raise SystemExit("Use --demo or provide --csv. See -h")

    res = run_pipeline(
        df, time_col, channels,
        win=args.win, baseline_len=args.baseline, k=args.k,
        norm=args.norm, thresh_method=args.method
    )

    # Optional: save PNG
    if args.save_prefix:
        out = f"{args.save_prefix}_sigma1_paper.png"
        res["figure"].savefig(out, dpi=150)
        print(f"[saved] {out}")

    # Always show the figure
    plt.show()

if __name__ == "__main__":
    main()
