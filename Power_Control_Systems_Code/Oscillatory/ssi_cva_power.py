# ssi_cva_power.py
# Paper-faithful stochastic subspace identification (SSI/CVA) for oscillatory modes.
# - Past/future block-Hankel, projections
# - CVA weighting: left whitening of Yf and thin-QR on Yp^T (orthonormalize past)
# - Observability, state sequences, LS estimate of Ad, C
# - Modes via bilinear transform; inter-area band filter; SISI = |alpha_c|
# - Conservative order selection with cap; de-duplicate conjugate pairs
# Usage (demo):
#   python ssi_cva_power.py --demo --fs 40 --minutes 4 --i 30 --j 1500

import argparse
from typing import Tuple
import numpy as np
import scipy.signal as sig
from numpy.linalg import svd, pinv

# ========== Utilities ==========

def block_hankel(y: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Build Y_{0|2i-1} with 2i block-rows and j columns from l-channel data y[t].
    y: (T, l)  -> H: (2*i*l, j)
    """
    T, l = y.shape
    assert 2 * i + j - 1 <= T, "Not enough samples for Hankel(2i x j)"
    H = np.zeros((2 * i * l, j), dtype=float)
    for r in range(2 * i):
        blk = y[r:r + j, :].T  # (l, j)
        H[r * l:(r + 1) * l, :] = blk
    return H

def split_past_future(H: np.ndarray, i: int, l: int) -> Tuple[np.ndarray, np.ndarray]:
    """From Y_{0|2i-1} return (Yp, Yf) = (Y_{0|i-1}, Y_{i|2i-1})."""
    Yp = H[0:i * l, :]
    Yf = H[i * l:2 * i * l, :]
    return Yp, Yf

def shifted_pf(y: np.ndarray, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (Yp^+, Yf^-) by shifting one sample forward (the 'shifted border').
    Returns shapes matching (Yp, Yf), except one fewer block-row effectively.
    """
    T, l = y.shape
    assert 2 * i + j <= T, "Not enough samples for shifted Hankel"
    Hs = block_hankel(y[1:], i, j)  # shift by 1 sample
    Yp_plus, Yf_minus = split_past_future(Hs, i, l)
    return Yp_plus, Yf_minus

def projection(Yf: np.ndarray, Yp: np.ndarray) -> np.ndarray:
    """
    Row-space projection O = Yf / Yp = Yf Yp^T (Yp Yp^T)^† Yp.
    """
    YYt = Yp @ Yp.T
    return Yf @ Yp.T @ pinv(YYt) @ Yp

def invsqrt_spd(M: np.ndarray) -> np.ndarray:
    """Return M^{-1/2} for SPD M via eigen-decomposition (robust whitening)."""
    d, U = np.linalg.eigh(M)
    d = np.clip(d, 1e-12, None)
    return U @ np.diag(d ** -0.5) @ U.T

def cva_svd(Yf: np.ndarray, Yp: np.ndarray, j: int):
    """
    CVA SVD using thin-QR on Yp^T to orthonormalize 'past':
      Yp^T = Qp Rp  (Qp: j×r, r ≤ i*l)
      O_q  = Yf @ Qp    ∈ R^{(i*l) × r}
      W1   = ((1/j) Yf Yf^T)^(-1/2)
    Do SVD:  W1 @ O_q = U S V^T
    Returns U, S, V, W1, Qp
    """
    Qp, Rp = np.linalg.qr(Yp.T, mode="reduced")  # Qp: j×r
    W1 = invsqrt_spd((Yf @ Yf.T) / j)
    O_q = Yf @ Qp
    U, S, Vt = svd(W1 @ O_q, full_matrices=False)
    return U, S, Vt.T, W1, Qp

def choose_order(S: np.ndarray, n_max: int, energy: float = 0.85, gap: bool = True) -> int:
    """
    Pick model order from singular values conservatively.
    - Try 'gap' (largest ratio drop) then clamp by energy criterion and n_max.
    """
    S = S[np.isfinite(S)]
    if S.size == 0:
        return 2
    n_guess = 2
    if gap and S.size >= 2:
        r = S[:-1] / np.maximum(S[1:], 1e-12)
        k = int(np.argmax(r)) + 1
        n_guess = max(2, min(k, n_max))
    cum = np.cumsum(S) / np.sum(S)
    n_energy = int(np.searchsorted(cum, energy)) + 1
    return max(2, min(n_guess, n_energy, n_max))

def build_observability(U1: np.ndarray, S1: np.ndarray, W1: np.ndarray, l: int):
    """Γ_i = W1^{-1} U1 S1^{1/2};  Γ_{i-1} = Γ_i without last l rows."""
    W1_inv = np.linalg.inv(W1)
    Gamma_i = W1_inv @ U1 @ np.diag(np.sqrt(S1))
    Gamma_i_minus = Gamma_i[:-l, :]
    return Gamma_i, Gamma_i_minus

def estimate_states(Gamma_i, Gamma_i_minus, Oi, Oi_minus):
    """
    X̂_i = Γ_i^† O_i,   X̂_{i+1} = Γ_{i-1}^† O_{i-1}
    Trim Oi_minus rows to match Γ_{i-1}.
    """
    Xi = pinv(Gamma_i) @ Oi
    rows = Gamma_i_minus.shape[0]
    Oi_minus_trim = Oi_minus[:rows, :]
    Xi1 = pinv(Gamma_i_minus) @ Oi_minus_trim
    return Xi, Xi1

def estimate_Ad_C(Xi, Xi1, Yf, l):
    """[A_d; C] = [ X̂_{i+1};  Y_{i|i} ]  X̂_i^†,  with Y_{i|i} top l rows of Y_f."""
    Yi_i = Yf[:l, :]
    M = np.vstack([Xi1, Yi_i])
    K = M @ pinv(Xi)
    n = Xi.shape[0]
    Ad = K[:n, :]
    C = K[n:, :]
    return Ad, C

def z_to_s(eigs_z: np.ndarray, Ts: float) -> np.ndarray:
    """Bilinear transform s = (2/Ts)*(z-1)/(z+1)."""
    return (2.0 / Ts) * (eigs_z - 1.0) / (eigs_z + 1.0)

def modes_from_Ad(Ad: np.ndarray, Ts: float):
    """Return (freq_Hz, zeta, alpha, beta, z_eigs, s_eigs) sorted by lowest damping."""
    z = np.linalg.eigvals(Ad)
    s = z_to_s(z, Ts)
    alpha = s.real
    beta = s.imag
    freq = np.abs(beta) / (2 * np.pi)
    zeta = -alpha / np.maximum(np.sqrt(alpha ** 2 + beta ** 2), 1e-12)
    order = np.argsort(zeta)  # least damped first
    return freq[order], zeta[order], alpha[order], beta[order], z[order], s[order]

def unique_conjugate_pairs(freq, zeta, alpha, beta, tol_f=1e-3, tol_z=1e-3):
    """
    Keep one representative from each conjugate pair (same freq & ζ within tol).
    Returns indices to keep.
    """
    keep = []
    used = np.zeros(len(freq), dtype=bool)
    for i in range(len(freq)):
        if used[i]:
            continue
        partner = None
        for j in range(i + 1, len(freq)):
            if used[j]:
                continue
            if abs(freq[i] - freq[j]) < tol_f and abs(zeta[i] - zeta[j]) < tol_z:
                partner = j
                break
        keep.append(i)
        if partner is not None:
            used[partner] = True
    return np.array(keep, dtype=int)

# ========== End-to-end SSI/CVA ==========

def ssi_cva(y, fs, i_blocks=40, j_cols=1200, band=(0.05, 2.0), n=None,
            cap_n=12, do_plots=False):
    """
    SSI/CVA on multichannel ambient outputs y[t,l].
    - Standardize channels, band-pass (0.05–2 Hz), downsample to ~10 Hz
    - Build Hankel (i,j), split past/future, shifted border
    - CVA via W1 whitening and thin-QR of Yp^T
    - Estimate Γ_i, states, Ad, C, modes; filter inter-area (0.1–2 Hz), stable
    - SISI = |alpha| of least-damped inter-area mode
    """
    y = np.asarray(y, float)
    T, l = y.shape

    # Standardize channels (z-score) to balance scales
    y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-9)

    # Filtering + downsample to ~10 Hz
    y_f = y.copy()
    if band is not None:
        wp = (np.array(band) / (fs / 2.0)).clip(1e-4, 0.999)
        b, a = sig.cheby1(4, 0.1, wp, btype='bandpass')
        y_f = sig.filtfilt(b, a, y_f, axis=0)
    if fs > 10.0:
        dec = int(round(fs / 10.0))
        y_f = sig.decimate(y_f, dec, axis=0, ftype='fir', zero_phase=True)
        fs = fs / dec

    Ts = 1.0 / fs

    # Hankel & splits
    H = block_hankel(y_f, i_blocks, j_cols)
    Yp, Yf = split_past_future(H, i_blocks, l)
    Yp_plus, Yf_minus = shifted_pf(y_f, i_blocks, j_cols)

    # Projections (row-space)
    Oi = projection(Yf, Yp)
    Oi_minus = projection(Yf_minus, Yp_plus)

    # CVA SVD with thin-QR on Yp^T (no dimension mismatch)
    U, S, V, W1, Qp = cva_svd(Yf, Yp, j_cols)

    # Order selection (conservative + cap)
    r = Qp.shape[1]                 # ≤ i*l
    n_cap = min(r, cap_n)
    if n is None:
        n = choose_order(S, n_max=n_cap, energy=0.85, gap=True)
    else:
        n = min(n, n_cap)

    U1 = U[:, :n]
    S1 = S[:n]

    # Observability & states
    Gamma_i, Gamma_i_minus = build_observability(U1, S1, W1, l)
    Xi, Xi1 = estimate_states(Gamma_i, Gamma_i_minus, Oi, Oi_minus)

    # System matrices
    Ad, C = estimate_Ad_C(Xi, Xi1, Yf, l)

    # Modes
    freq, zeta, alpha, beta, z_eigs, s_eigs = modes_from_Ad(Ad, Ts)

    # Inter-area band and stability filter; SISI
    band_lo, band_hi = 0.1, 2.0  # Hz
    ok = (freq >= band_lo) & (freq <= band_hi) & (zeta >= 0.0)
    if np.any(ok):
        # sort by lowest damping, then keep unique conjugate pairs
        idx = np.argsort(zeta[ok])
        freq, zeta, alpha, beta = freq[ok][idx], zeta[ok][idx], alpha[ok][idx], beta[ok][idx]
        keep = unique_conjugate_pairs(freq, zeta, alpha, beta)
        freq, zeta, alpha, beta = freq[keep], zeta[keep], alpha[keep], beta[keep]
        SISI = float(np.abs(alpha[0]))  # least-damped inter-area mode
    else:
        SISI = float(np.min(np.abs(alpha)))  # fallback (may be spurious)

    # Optional plots
    if do_plots:
        import matplotlib.pyplot as plt
        def plot_singulars(S):
            plt.figure(figsize=(6, 3))
            plt.semilogy(S, marker='o')
            plt.title("CVA singular values"); plt.xlabel("index"); plt.ylabel("σ (log)")
            plt.tight_layout(); plt.show()
        def plot_modes(freq, zeta):
            plt.figure(figsize=(5, 4))
            plt.scatter(freq, zeta)
            plt.xlim(0, 2.0); plt.xlabel("Frequency (Hz)")
            plt.ylim(0, 0.3); plt.ylabel("Damping ratio ζ")
            plt.title("Identified inter-area modes")
            plt.grid(True, ls='--', alpha=0.5)
            plt.tight_layout(); plt.show()
        plot_singulars(S)
        if len(freq) > 0:
            plot_modes(freq, zeta)

    return {
        "Ad": Ad, "C": C, "freq_Hz": freq, "zeta": zeta, "alpha": alpha, "beta": beta,
        "z_eigs": z_eigs, "s_eigs": s_eigs, "SISI": SISI, "order": n, "fs_used": fs,
        "singulars": S
    }

# ========== Demo data & CLI ==========

def make_ambient_demo(T=4*60*40, fs=40.0, modes=[(0.35, 0.06), (0.75, 0.12)], l=2,
                      snr_db=20, seed=1):
    """
    Multi-output ambient response with two underdamped modes (f, ζ).
    Returns y[t,l], fs.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(T)) / fs
    y = np.zeros((len(t), l))
    for f, zeta in modes:
        wn = 2 * np.pi * f
        A = np.array([[0, 1], [-wn ** 2, -2 * zeta * wn]])
        dt = 1.0 / fs
        I = np.eye(2)
        M = np.linalg.inv(I - 0.5 * A * dt) @ (I + 0.5 * A * dt)  # bilinear
        x = np.zeros(2)
        w = rng.normal(scale=1.0, size=len(t))
        for k in range(len(t)):
            x = M @ x + np.array([0, 1]) * w[k] * np.sqrt(dt)
            y[k, 0] += x[0]
            y[k, 1] += 0.7 * x[0] + 0.3 * rng.normal()
    # color & add noise at target SNR
    y = sig.lfilter([1, 0, -0.85], [1], y, axis=0)
    p_sig = np.mean(y ** 2, axis=0)
    snr = 10 ** (snr_db / 10)
    noise_var = p_sig / snr
    y += rng.normal(scale=np.sqrt(noise_var), size=y.shape)
    return y, fs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run synthetic ambient demo")
    ap.add_argument("--fs", type=float, default=40.0, help="input sampling rate (Hz)")
    ap.add_argument("--minutes", type=float, default=4.0, help="length of record (min)")
    ap.add_argument("--i", type=int, default=30, help="block-rows per past/future (i)")
    ap.add_argument("--j", type=int, default=1500, help="columns (j)")
    ap.add_argument("--order", type=int, default=None, help="force model order n; else auto")
    ap.add_argument("--plots", action="store_true", help="show singulars/modes plots")
    args = ap.parse_args()

    if args.demo:
        T = int(args.minutes * 60 * args.fs)
        y, fs = make_ambient_demo(T=T, fs=args.fs)
        out = ssi_cva(y, fs, i_blocks=args.i, j_cols=args.j, n=args.order, do_plots=args.plots)
        print(f"[SSI] order n = {out['order']}, fs_used={out['fs_used']:.2f} Hz")
        if len(out["freq_Hz"]) > 0:
            print(f"[SSI] SISI (|alpha_c|) = {out['SISI']:.4f} s^-1")
            print("Top inter-area modes (unique conjugate pairs):")
            for f, z in zip(out["freq_Hz"][:6], out["zeta"][:6]):
                print(f"  f={f:.3f} Hz, ζ={z:.3f}")
        else:
            print("[SSI] No stable inter-area modes found in 0.1–2.0 Hz; SISI fallback used.")
    else:
        print("Use --demo, or load your PMU CSV (y: T×l, fs) and call ssi_cva(y, fs, ...).")

if __name__ == "__main__":
    main()
