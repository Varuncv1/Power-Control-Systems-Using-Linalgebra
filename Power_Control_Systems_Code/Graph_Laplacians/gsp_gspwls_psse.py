# Create a runnable Python module that implements the DC GSP‑WLS estimator,
# missing‑data reconstruction, and the greedy sensor selection (Algorithm 1)
# from Dabush et al. (Sensors 2023). Includes a small demo on a synthetic
# Laplacian to show usage.
#
# You can plug your own Y‑bus (or B=L) in place of the synthetic test.
from __future__ import annotations
import numpy as np
import numpy.linalg as npl
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional

# --------------------------- Core math ---------------------------

def remove_ref(mat: np.ndarray, ref: int) -> np.ndarray:
    """Remove row and column 'ref' from a square matrix."""
    idx = [i for i in range(mat.shape[0]) if i != ref]
    return mat[np.ix_(idx, idx)]

def selector(rows: Iterable[int], n: int) -> np.ndarray:
    """Return a row-selector matrix S so that S @ a == a[rows]."""
    S = np.zeros((len(rows), n))
    for r, i in enumerate(rows):
        S[r, i] = 1.0
    return S

def _as_Rinv(RS, m: int):
    """Return an m*m inverse covariance matrix from RS (scalar or array)."""
    if np.isscalar(RS):
        return (1.0 / float(RS)) * np.eye(m)
    RS = np.asarray(RS)
    if RS.ndim == 1:
        # diagonal specified as a vector
        d = RS[:m]
        return np.diag(1.0 / d)
    # square matrix case; clip or pad as needed
    if RS.shape != (m, m):
        # take top-left or rebuild diagonal from its diag
        d = np.diag(RS)
        d = d[:m] if d.size >= m else np.pad(d, (0, m - d.size), mode="edge")
        return np.diag(1.0 / d)
    # exactly m×m
    return npl.inv(RS)

def _as_R(RS, m: int):
    """Return an m*m covariance matrix from RS (scalar or array), for CRB."""
    if np.isscalar(RS):
        return float(RS) * np.eye(m)
    RS = np.asarray(RS)
    if RS.ndim == 1:
        d = RS[:m]
        return np.diag(d)
    if RS.shape != (m, m):
        d = np.diag(RS)
        d = d[:m] if d.size >= m else np.pad(d, (0, m - d.size), mode="edge")
        return np.diag(d)
    return RS

# --------------------------- Estimators ---------------------------

def gsp_wls_dc(B: np.ndarray, S_rows: List[int], zS: np.ndarray, RS: np.ndarray,
               mu: float, ref: int = 0) -> np.ndarray:
    """
    DC model with bus injections only: z_bus = B @ theta + e.
    Partial measurements: zS = H_S,V @ theta + eS with H_S,V = selector(S_rows, N) @ B.

    Implements Eq. (26)-(27): θ̂_{V̄} = (H^T R^{-1} H + μ L_{V̄})^{-1} H^T R^{-1} zS,
    with L = B and θ_ref = 0.
    """
    N = B.shape[0]
    # Measurement matrix for selected buses: pick those rows of B
    H = selector(S_rows, N) @ B  # shape (|S|, N)
    # Remove reference column from H, and ref row/col from L
    idxV = [i for i in range(N) if i != ref]
    Hvr = H[:, idxV]
    Lv = remove_ref(B, ref)

    Rinv = npl.inv(RS)
    A = Hvr.T @ Rinv @ Hvr + mu * Lv
    b = Hvr.T @ Rinv @ zS
    thetav = npl.solve(A, b)  # θ without reference
    theta = np.zeros(N); theta[idxV] = thetav; theta[ref] = 0.0
    return theta

def wls_dc(B: np.ndarray, S_rows: List[int], zS: np.ndarray, RS: np.ndarray,
           ref: int = 0) -> np.ndarray:
    """
    Classical WLS for observable case (μ=0). If not observable, this will raise.
    """
    return gsp_wls_dc(B, S_rows, zS, RS, mu=0.0, ref=ref)

def reconstruct_missing(B: np.ndarray, theta_hat: np.ndarray, missing_rows: List[int]) -> np.ndarray:
    """Eq. (33): ẑ_{missing} = H_{missing} θ̂ with H_{missing} = selector(rows,N) @ B."""
    Hmiss = selector(missing_rows, B.shape[0]) @ B
    return Hmiss @ theta_hat

# --------------------------- Sensor selection (Algorithm 1) ---------------------------

def Ktilde(B: np.ndarray, S_rows: List[int], RS, mu: float, ref: int = 0) -> np.ndarray:
    """Return K̃(S, μ) mapping z_S -> θ_{V\{ref}}."""
    N = B.shape[0]
    H = selector(S_rows, N) @ B                 # (|S| × N)
    idxV = [i for i in range(N) if i != ref]
    Hvr = H[:, idxV]                            # (|S| × (N-1))
    Lv = remove_ref(B, ref)                     # ((N-1) × (N-1))
    m = len(S_rows)
    Rinv = _as_Rinv(RS, m)                      # (|S| × |S|)
    A = Hvr.T @ Rinv @ Hvr + mu * Lv            # ((N-1) × (N-1))
    return npl.solve(A, Hvr.T @ Rinv)           # ((N-1) × |S|)

def crb_trace(B: np.ndarray, S_rows: List[int], RS, mu: float, ref: int = 0) -> float:
    """CRB(S) = Tr( K̃ R_S K̃ᵀ )."""
    m = len(S_rows)
    K = Ktilde(B, S_rows, RS, mu, ref)          # ((N-1) × m)
    Rm = _as_R(RS, m)                            # (m × m)
    return float(np.trace(K @ Rm @ K.T))

def greedy_select(B: np.ndarray, q: int, RS, mu: float, ref: int = 0,
                  start: Optional[List[int]] = None) -> List[int]:
    """
    Greedy selection minimizing CRB trace. RS can be:
      - scalar variance (σ²), or
      - diagonal vector, or
      - square covariance matrix (will be adapted to |S|).
    """
    N = B.shape[0]
    S = [] if start is None else list(start)
    candidates = [i for i in range(N) if i not in S]
    for _ in range(q - len(S)):
        best_i, best_val = None, np.inf
        for i in candidates:
            rows = S + [i]
            val = crb_trace(B, rows, RS, mu, ref)
            if val < best_val:
                best_val, best_i = val, i
        S.append(best_i)
        candidates.remove(best_i)
    return S

# --------------------------- Synthetic demo ---------------------------

def make_ring_laplacian(N: int, w: float = 1.0) -> np.ndarray:
    """Simple connected graph Laplacian (cycle)."""
    L = np.zeros((N, N))
    for i in range(N):
        j = (i + 1) % N
        L[i, i] += w; L[j, j] += w
        L[i, j] -= w; L[j, i] -= w
    return L

def sample_smooth_theta(L: np.ndarray, rng=None, alpha: float = 1e-2) -> np.ndarray:
    """
    Produce a smooth state by filtering white noise with (L + αI)^{-1}.
    Enforces θ_ref = 0 at the end.
    """
    rng = np.random.default_rng(rng)
    N = L.shape[0]
    w = rng.normal(0, 1, N)
    theta = npl.solve(L + alpha * np.eye(N), w)
    theta -= theta[0]  # set ref bus 0 to zero
    return theta

def make_grid_laplacian(nr=5, nc=6, w=1.0):
    N = nr*nc
    L = np.zeros((N,N))
    def idx(r,c): return r*nc + c
    for r in range(nr):
        for c in range(nc):
            i = idx(r,c)
            if r+1<nr:
                j = idx(r+1,c); L[i,i]+=w; L[j,j]+=w; L[i,j]-=w; L[j,i]-=w
            if c+1<nc:
                j = idx(r,c+1); L[i,i]+=w; L[j,j]+=w; L[i,j]-=w; L[j,i]-=w
    return L


def demo_dc():
    """
    DC demo for GSP-WLS state estimation.
    Shows effect of sensor placement and Laplacian regularization.
    """

    rng = np.random.default_rng(0)
    N = 30

    # --- Choose graph topology ---
    # B = make_ring_laplacian(N)
    B = make_grid_laplacian(5, 6)  # 5x6 grid (uncomment for nicer results)

    # --- True state and noisy injections ---
    theta_true = sample_smooth_theta(B, rng)
    sigma2 = 1e-3
    z_full = B @ theta_true + rng.normal(0, np.sqrt(sigma2), B.shape[0])

    # --- Config ---
    q = 12
    mu = 1.0

    # --- Greedy with seed spread (break ring symmetry) ---
    S0 = [0, N // 2]
    S = greedy_select(B, q=q, RS=sigma2, mu=mu, ref=0, start=S0)
    zS = z_full[S]
    RS_mat = sigma2 * np.eye(len(S))

    # --- Estimation (with ridge) ---
    def gsp_estimate(B, S, zS, RS, mu, ref=0, eps=1e-8):
        N = B.shape[0]
        H = selector(S, N) @ B
        idxV = [i for i in range(N) if i != ref]
        Hvr = H[:, idxV]
        Lv = remove_ref(B, ref)
        Rinv = np.linalg.inv(RS)
        A = Hvr.T @ Rinv @ Hvr + mu * Lv + eps * np.eye(N - 1)
        b = Hvr.T @ Rinv @ zS
        thetav = np.linalg.solve(A, b)
        theta = np.zeros(N); theta[idxV] = thetav
        return theta

    theta_hat = gsp_estimate(B, S, zS, RS_mat, mu)

    # --- Metrics ---
    missing = [i for i in range(N) if i not in S]
    zhat_missing = reconstruct_missing(B, theta_hat, missing)
    err = np.linalg.norm(theta_hat - theta_true) / np.linalg.norm(theta_true)
    mse_zmiss = np.mean((B[missing, :] @ theta_true - zhat_missing) ** 2)

    # Residual on measured injections
    H = selector(S, N) @ B
    res = zS - H @ theta_hat

    print(f"[DC] Selected buses (q={q}):", S)
    print(f"[DC] Relative state error (mu={mu}): {err:.3e}")
    print(f"[DC] MSE missing z: {mse_zmiss:.3e}")
    print(f"[DC] ||residual_on_measured|| = {np.linalg.norm(res):.3e}")

    # --- Sanity: all sensors ---
    S_all = list(range(N))
    z_all = z_full.copy()
    RS_all = sigma2 * np.eye(len(S_all))
    theta_all = gsp_estimate(B, S_all, z_all, RS_all, mu=1e-6)
    err_all = np.linalg.norm(theta_all - theta_true) / np.linalg.norm(theta_true)
    print(f"[DC] Sanity (all sensors): rel err = {err_all:.3e}")



if __name__ == "__main__":
    demo_dc()
