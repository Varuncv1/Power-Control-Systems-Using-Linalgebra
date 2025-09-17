# pmu_wls_generator_params.py
# WLS/LS estimation of synchronous generator parameters from PMU-like data
# Case 1 (Bander et al.): ΔPe → Δω  (D≈0). ZOH and Tustin supported.
# ZOH path now matches the paper exactly (indexing and recovery formulas).

import argparse
import numpy as np
import pandas as pd
from numpy.linalg import pinv, lstsq
from scipy import signal

# ---------------------------
# Regressors / solvers
# ---------------------------

def fit_arx_zoh(y, u):
    """
    Paper ZOH ARX (Case 1):
      y[k+2] = -a1*y[k+1] - a0*y[k] + b1*u[k+1] + b0*u[k]
    Returns a1, a0, b1, b0 (LS solution).
    """
    y = np.asarray(y).ravel()
    u = np.asarray(u).ravel()
    N = len(y)
    rows = []
    rhs  = []
    for k in range(N - 2):
        rows.append([-y[k+1], -y[k], u[k+1], u[k]])
        rhs.append(y[k+2])
    A = np.asarray(rows, float)
    b = np.asarray(rhs,  float)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    a1, a0, b1, b0 = x
    return float(a1), float(a0), float(b1), float(b0)

def arx_regressor_tustin(y, u):
    """
    Tustin ARX used in the paper for Case 1:
      y[k] = -a1 y[k-1] - a0 y[k-2] + b2 u[k] + b1 u[k-1] + b0 u[k-2]
    Returns A, b for LS/WLS.
    """
    y = np.asarray(y).ravel()
    u = np.asarray(u).ravel()
    k0 = 2
    N = len(y)
    rows = N - k0
    A = np.zeros((rows, 5))
    bvec = np.zeros(rows)
    r = 0
    for k in range(k0, N):
        A[r, 0] = -y[k-1]  # a1
        A[r, 1] = -y[k-2]  # a0
        A[r, 2] =  u[k]    # b2
        A[r, 3] =  u[k-1]  # b1
        A[r, 4] =  u[k-2]  # b0
        bvec[r]  =  y[k]
        r += 1
    return A, bvec

def solve_wls(A, b, w=None):
    """
    Weighted least squares: minimize ||W^(1/2)(Ax-b)||_2.
    If w is None → ordinary LS.
    """
    if w is None:
        x, _, _, _ = lstsq(A, b, rcond=None)
        resid = b - A @ x
        dof = max(1, A.shape[0] - A.shape[1])
        sigma2 = float(resid.T @ resid) / dof
        cov = sigma2 * pinv(A.T @ A)
        return x, resid, cov, sigma2
    w = np.asarray(w).ravel()
    Wsqrt = np.sqrt(w)[:, None]
    A_w = Wsqrt * A
    b_w = (Wsqrt[:, 0] * b)
    x, _, _, _ = lstsq(A_w, b_w, rcond=None)
    resid = b - A @ x
    dof = max(1, A.shape[0] - A.shape[1])
    sigma2 = float((resid**2 @ w)) / dof
    # simple covariance approx (uses unweighted normal equations)
    cov = sigma2 * pinv(A.T @ A)
    return x, resid, cov, sigma2

# ---------------------------
# Parameter recovery – ZOH (paper-faithful)
# ---------------------------

def recover_params_zoh(a1, a0, b1, b0, h):
    """
    Paper eqs. for Case 1 (ΔPe→Δω), ZOH:
      T = -h / ln(a0)
      R = (b1+b0) / (1 + e^{-h/T} + a1)
      ω = (1/h) * acos( (-a1 * e^{h/(2T)}) / 2 )
      H = 2T / ( R * (1 + 4 T^2 ω^2) )
    """
    if not (0 < a0 < 1):
        raise ValueError("a0 must be in (0,1) for a stable underdamped case")
    T = -h / np.log(a0)
    R = (b1 + b0) / (1.0 + np.exp(-h / T) + a1)
    arg = (-a1 * np.exp(h / (2.0 * T))) / 2.0
    arg = np.clip(arg, -1.0, 1.0)
    omega = np.arccos(arg) / h
    H = (2.0 * T) / (R * (1.0 + 4.0 * (T**2) * (omega**2)))
    return H, R, T, omega

# ---------------------------
# Parameter recovery – Tustin (kept from prior version)
# ---------------------------

def recover_params_tustin(a1, a0, b2, b1, b0, h):
    """
    Heuristic recovery for Case 1 with Tustin. Keeps earlier approach.
    """
    k = 2.0 / h
    def H_from_R(Rguess, T):
        return (1 + T*k - b2/Rguess) / (2*b2*k*(T*k + 1))
    # T from b2/b1 (paper derivation)
    B = b2 / (b1 if abs(b1) > 1e-12 else 1e-12)
    T = 2.0 / (k * B) if B != 0 else 1.0
    def a1_from_R(Rguess):
        H = H_from_R(Rguess, T)
        alpha = 2*H*Rguess*T*(k**2) + 2*H*Rguess*k + 1
        return (2 - 4*H*Rguess*T*(k**2)) / alpha
    Rguess = 0.05
    for _ in range(30):
        f = a1_from_R(Rguess) - a1
        dR = 1e-4 * max(1.0, abs(Rguess))
        df = (a1_from_R(Rguess + dR) - a1_from_R(Rguess - dR)) / (2*dR)
        step = f/df if df != 0 else 0.0
        Rnext = Rguess - step
        if not np.isfinite(Rnext) or Rnext <= 1e-6:
            Rnext = max(1e-4, Rguess*0.5)
        if abs(Rnext - Rguess) < 1e-8:
            break
        Rguess = Rnext
    R = Rguess
    H = H_from_R(R, T)
    return H, R, T

# ---------------------------
# Synthetic data generator (Case 1)
# ---------------------------

def synth_pe_to_omega(H=2.5, R=0.05, T=0.5, h=0.1, N=800, step_at=100, step=0.2, noise_std=1e-4):
    """
    Generate (u=ΔPe, y=Δω) by simulating:
      G(s) = (T s + 1) / (2 H T s^2 + 2 H s + 1/R)   (D≈0 per paper Case 1)
    Discretization: ZOH with sampling h. Input: step in ΔPe at step_at.
    """
    # Continuous-time TF
    num = [T, 1.0]
    den = [2*H*T, 2*H, 1.0/R]
    # TF → SS, then ZOH discretize (returns 5-tuple)
    A, B, C, D = signal.tf2ss(num, den)
    Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), h, method='zoh')

    # Simulate
    x = np.zeros(Ad.shape[0])
    u = np.zeros(N); u[step_at:] = step
    y = np.zeros(N)
    for k in range(N):
        y[k] = (Cd @ x + Dd * u[k]).item()
        x = (Ad @ x + Bd.flatten() * u[k])
    if noise_std > 0:
        y += np.random.normal(0.0, noise_std, size=N)
    return u, y

# ---------------------------
# Estimation flows
# ---------------------------

def estimate_case1_pe_to_omega(u, y, h, method='zoh', weights=None):
    """
    Estimate H, R, T (and ω for ZOH) from (u=ΔPe, y=Δω).
    """
    if method == 'zoh':
        a1, a0, b1, b0 = fit_arx_zoh(y, u)
        H, R, T, omega = recover_params_zoh(a1, a0, b1, b0, h)
        # Build LS stats using the same regression used to get a's/b's
        # (recreate A,b to compute residuals/cov)
        rows, rhs = [], []
        for k in range(len(y)-2):
            rows.append([-y[k+1], -y[k], u[k+1], u[k]])
            rhs.append(y[k+2])
        A = np.asarray(rows); b = np.asarray(rhs)
        x = np.array([a1, a0, b1, b0])
        resid = b - A @ x
        dof = max(1, A.shape[0]-A.shape[1])
        sigma2 = float(resid.T @ resid) / dof
        cov = sigma2 * pinv(A.T @ A)
        return {"H": H, "R": R, "T": T, "omega": omega,
                "coeffs": dict(a1=a1, a0=a0, b1=b1, b0=b0),
                "sigma2": sigma2, "resid": resid, "cov": cov}
    elif method == 'tustin':
        A, b = arx_regressor_tustin(y, u)
        x, resid, cov, sigma2 = solve_wls(A, b, w=weights)
        a1, a0, b2, b1, b0 = x
        H, R, T = recover_params_tustin(a1, a0, b2, b1, b0, h)
        return {"H": H, "R": R, "T": T,
                "coeffs": dict(a1=float(a1), a0=float(a0), b2=float(b2), b1=float(b1), b0=float(b0)),
                "sigma2": sigma2, "resid": resid, "cov": cov}
    else:
        raise ValueError("method must be 'zoh' or 'tustin'")

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="WLS/LS estimation of generator params from PMU (Case 1: ΔPe→Δω).")
    ap.add_argument("--demo", action="store_true", help="run synthetic demo and estimate")
    ap.add_argument("--H", type=float, default=2.5)
    ap.add_argument("--R", type=float, default=0.05)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--h", type=float, default=0.1)
    ap.add_argument("--N", type=int, default=800)
    ap.add_argument("--step_at", type=int, default=100)
    ap.add_argument("--step", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=1e-4)
    ap.add_argument("--method", choices=["zoh", "tustin"], default="zoh")
    ap.add_argument("--csv", type=str, help="CSV with columns for input/output")
    ap.add_argument("--ucol", type=str, default="u", help="CSV column for ΔPe")
    ap.add_argument("--ycol", type=str, default="y", help="CSV column for Δω (or Δδ)")
    ap.add_argument("--weights", type=str, help="CSV column with weights (optional)")
    ap.add_argument("--plot", action="store_true", help="overlay ARX response vs data")
    args = ap.parse_args()

    if args.demo:
        u, y = synth_pe_to_omega(H=args.H, R=args.R, T=args.T, h=args.h,
                                 N=args.N, step_at=args.step_at, step=args.step, noise_std=args.noise)
        out = estimate_case1_pe_to_omega(u, y, h=args.h, method=args.method, weights=None)
        print(f"[Demo true]   H={args.H:.3f}, R={args.R:.5f}, T={args.T:.3f}, h={args.h:.3f}")
        if args.method == "zoh":
            print(f"[ZOH  est ]  H={out['H']:.3f}, R={out['R']:.5f}, T={out['T']:.3f}, ω={out['omega']:.3f} rad/s")
            print("Coeffs:", {k: float(v) for k, v in out["coeffs"].items()})
        else:
            print(f"[TUSTIN est] H={out['H']:.3f}, R={out['R']:.5f}, T={out['T']:.3f}")
            print("Coeffs:", {k: float(v) for k, v in out["coeffs"].items()})
        print(f"sigma^2(resid) ≈ {out['sigma2']:.3e}")

        if args.plot:
            import matplotlib.pyplot as plt
            # Build ARX prediction with estimated coeffs
            if args.method == "zoh":
                a1, a0, b1, b0 = (out["coeffs"][k] for k in ("a1","a0","b1","b0"))
                yhat = np.zeros_like(y)
                for k in range(len(y)-2):
                    yhat[k+2] = -a1*y[k+1] - a0*y[k] + b1*u[k+1] + b0*u[k]
            else:
                a1, a0, b2, b1, b0 = (out["coeffs"][k] for k in ("a1","a0","b2","b1","b0"))
                yhat = np.zeros_like(y)
                for k in range(2, len(y)):
                    yhat[k] = -a1*y[k-1] - a0*y[k-2] + b2*u[k] + b1*u[k-1] + b0*u[k-2]
            t = np.arange(len(y))*args.h
            plt.figure(figsize=(8,3))
            plt.plot(t, y, label="measured Δω")
            plt.plot(t, yhat, '--', label="ARX fit")
            plt.legend(); plt.xlabel("time (s)"); plt.ylabel("Δω (pu)")
            plt.title("Measured vs ARX (estimated)")
            plt.grid(ls='--', alpha=0.6); plt.tight_layout(); plt.show()
        return

    if args.csv:
        df = pd.read_csv(args.csv)
        u = df[args.ucol].to_numpy(float)
        y = df[args.ycol].to_numpy(float)
        w = df[args.weights].to_numpy(float) if args.weights and args.weights in df.columns else None
        out = estimate_case1_pe_to_omega(u, y, h=args.h, method=args.method, weights=w)
        print(f"[{args.method.upper()} est] H={out['H']:.3f}, R={out['R']:.5f}, T={out['T']:.3f}")
        print("Coeffs:", {k: float(v) for k, v in out["coeffs"].items()})
        print(f"sigma^2(resid) ≈ {out['sigma2']:.3e}")
        return

    print("Nothing to do. Use --demo or --csv. See --help.")

if __name__ == "__main__":
    main()
