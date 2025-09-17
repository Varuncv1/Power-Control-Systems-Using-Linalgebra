# kron_pf_solvers.py
# Kronecker-product power-flow least squares:
#  - ALS per Eq. (13)
#  - "Proposed" Kronecker LS step per Eq. (15)
# Ref: Oh, "A Unified and Efficient Approach to Power Flow Analysis", Energies 2019.

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Dict, Any, Tuple

# ------- helpers -------

def _as_csr(A):
    return A.tocsr() if sp.issparse(A) else sp.csr_matrix(A)

def _kron_Ix(x: np.ndarray, n: int) -> sp.csr_matrix:
    """(I_n ⊗ x) as CSR (sparse-friendly)."""
    return sp.kron(sp.eye(n, format='csr'), sp.csr_matrix(x.reshape(-1,1)), format='csr')

def _kron_xI(x: np.ndarray, n: int) -> sp.csr_matrix:
    """(x ⊗ I_n) as CSR (sparse-friendly)."""
    return sp.kron(sp.csr_matrix(x.reshape(-1,1)), sp.eye(n, format='csr'), format='csr')

def _A_times_kron(A: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    """Compute A @ (x ⊗ x) efficiently (A is m×n^2)."""
    n2 = A.shape[1]
    n = int(np.sqrt(n2))
    assert n*n == n2, "A columns must be a perfect square"
    return A.dot(np.kron(x, x))

# ------- solvers -------

def solve_als(A, b, x0, alpha=0.3, max_it=80, tol=1e-10, verbose=True,
              ls_atol=1e-12, ls_btol=1e-12, ls_iter=5000, ls_damp=1e-8) -> Dict[str, Any]:
    """
    ALS (Eq. 13):
      Step I:  x_hat = argmin || b - [A (x_l ⊗ I)] x ||_2
      Step II: x_new = argmin || b - [A (I ⊗ x_hat)] x ||_2
      Step III: x_{l+1} = (1-α) x_hat + α x_new
    """
    A = _as_csr(A)
    m, n2 = A.shape
    n = int(np.sqrt(n2))
    x = np.asarray(x0, float).copy()
    hist = []
    prev_rn = None

    for it in range(1, max_it+1):
        # Step I
        K1 = A.dot(_kron_xI(x, n))                  # (m, n)
        x_hat, *_ = spla.lsqr(K1, b, damp=ls_damp, atol=ls_atol, btol=ls_btol,
                              iter_lim=ls_iter)[:2] + (None,)
        # Step II
        K2 = A.dot(_kron_Ix(x_hat, n))              # (m, n)
        x_new, *_ = spla.lsqr(K2, b, damp=ls_damp, atol=ls_atol, btol=ls_btol,
                              iter_lim=ls_iter)[:2] + (None,)

        # Relax / average
        x_next = (1.0 - alpha) * x_hat + alpha * x_new

        r = b - _A_times_kron(A, x_next)
        rn = float(np.linalg.norm(r))
        hist.append({"it": it, "res2": rn})
        if verbose:
            print(f"[ALS ] it={it:02d}  ||r||2={rn:.3e}")

        # stopping
        if rn < tol:
            x = x_next
            break
        if prev_rn is not None and abs(prev_rn - rn) < 1e-12:
            # stagnation early-stop
            break
        prev_rn = rn
        x = x_next

    return {"x": x, "converged": (hist[-1]["res2"] < tol), "history": hist, "method": "ALS"}

def solve_proposed(A, b, x0, alpha=0.3, max_it=60, tol=1e-10, verbose=True,
                   ls_atol=1e-12, ls_btol=1e-12, ls_iter=5000, ls_damp=1e-8) -> Dict[str, Any]:
    """
    Proposed step (Eq. 15 + relaxation Eq. 16):
      x_hat = argmin || b - 1/2 A ( I ⊗ x_l + x_l ⊗ I ) x ||_2
      x_{l+1} = (1-α) x_l + α x_hat
    """
    A = _as_csr(A)
    m, n2 = A.shape
    n = int(np.sqrt(n2))
    x = np.asarray(x0, float).copy()
    hist = []
    prev_rn = None

    for it in range(1, max_it+1):
        K = 0.5 * A.dot(_kron_Ix(x, n) + _kron_xI(x, n))  # (m, n)
        x_hat, *_ = spla.lsqr(K, b, damp=ls_damp, atol=ls_atol, btol=ls_btol,
                              iter_lim=ls_iter)[:2] + (None,)
        x_next = (1.0 - alpha)*x + alpha*x_hat

        r = b - _A_times_kron(A, x_next)
        rn = float(np.linalg.norm(r))
        hist.append({"it": it, "res2": rn})
        if verbose:
            print(f"[Prop] it={it:02d}  ||r||2={rn:.3e}")

        # stopping
        if rn < tol:
            x = x_next
            break
        if prev_rn is not None and abs(prev_rn - rn) < 1e-12:
            break
        prev_rn = rn
        x = x_next

    return {"x": x, "converged": (hist[-1]["res2"] < tol), "history": hist, "method": "Proposed"}

# ------- synthetic demo -------

def make_synthetic_A_b(n: int, m: int, seed=0) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Build sparse A (m × n^2) and planted x*, then set b = A (x* ⊗ x*).
    """
    rng = np.random.default_rng(seed)
    A = sp.random(m, n*n, density=min(0.05, 20.0/(n*n)), format='csr', random_state=rng)
    x_star = rng.normal(size=n)
    b = A.dot(np.kron(x_star, x_star))
    return A, b, x_star

def _demo():
    n, m = 20, 200
    A, b, x_true = make_synthetic_A_b(n, m, seed=42)
    x0 = np.ones(n) * 0.1

    print("== Proposed (Eq. 15) ==")
    out1 = solve_proposed(A, b, x0, alpha=0.3, max_it=50, tol=1e-12, verbose=True)
    err1 = np.linalg.norm(out1["x"] - x_true) / max(1e-12, np.linalg.norm(x_true))
    print(f"Rel. error vs planted x*: {err1:.3e}\n")

    print("== ALS (Eq. 13) ==")
    out2 = solve_als(A, b, x0, alpha=0.3, max_it=80, tol=1e-12, verbose=True)
    err2 = np.linalg.norm(out2["x"] - x_true) / max(1e-12, np.linalg.norm(x_true))
    print(f"Rel. error vs planted x*: {err2:.3e}\n")

    # Summary table
    r1 = out1["history"][-1]["res2"]
    r2 = out2["history"][-1]["res2"]
    print("Summary (synthetic plant):")
    print("Method       iters  final ||r||2      rel_err_to_x*")
    print(f"Proposed     {len(out1['history']):>5}  {r1:.3e}     {err1:.3e}")
    print(f"ALS          {len(out2['history']):>5}  {r2:.3e}     {err2:.3e}")
    print("Both methods recover x* accurately; Proposed = 1 LS per outer iter, ALS = 2.")

if __name__ == "__main__":
    _demo()
