import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la


# ---------------------------
# Pencil builders (Explicit & Semi-Implicit DAEs)
# ---------------------------

def build_pencil_explicit(fx, fy, gx, gy):
    """
    Explicit DAEs:
      x' = f(x,y),   0 = g(x,y)
    Linearized at equilibrium (x0,y0):
      Δx' = fx Δx + fy Δy
      0   = gx Δx + gy Δy
    Primal GEP pencil:  (A - λ B) v = 0  with
      A = [[fx, fy],
           [gx, gy]]
      B = [[I,  0 ],
           [0,  0 ]]
    """
    fx, fy, gx, gy = map(sp.csr_matrix, (fx, fy, gx, gy))
    p, q = fx.shape[0], gy.shape[0]
    A = sp.bmat([[fx, fy],
                 [gx, gy]], format="csr")
    I = sp.eye(p, format="csr")
    Zpx = sp.csr_matrix((p, q))
    Zqp = sp.csr_matrix((q, p))
    Zqq = sp.csr_matrix((q, q))
    B = sp.bmat([[I,   Zpx],
                 [Zqp, Zqq]], format="csr")
    return A, B

def build_pencil_semiimplicit(fx_t, fy_t, gx_t, gy_t, T, R):
    """
    Semi-Implicit DAEs (as in paper):
      T x' = f̃(x,y)
      R x' = g̃(x,y)
    Linearized:
      T Δx' = fx_t Δx + fy_t Δy
      R Δx' = gx_t Δx + gy_t Δy
    Primal GEP pencil:
      Ã = [[fx_t, fy_t],
            [gx_t, gy_t]]
      B̃ = [[T,    0   ],
            [R,    0   ]]
    """
    fx_t, fy_t, gx_t, gy_t, T, R = map(sp.csr_matrix, (fx_t, fy_t, gx_t, gy_t, T, R))
    A = sp.bmat([[fx_t, fy_t],
                 [gx_t, gy_t]], format="csr")
    p, q = T.shape[0], gy_t.shape[0]
    Zpx = sp.csr_matrix((p, q))
    Zqp = sp.csr_matrix((q, p))
    Zqq = sp.csr_matrix((q, q))
    B = sp.bmat([[T,   Zpx],
                 [R,   Zqq]], format="csr")
    return A, B

# ---------------------------
# Generalized eigen solve (shift–invert)
# ---------------------------

def solve_gevp(A, B, k=8, sigma=0.0):
    """
    Solve (A - λ B)v = 0. Uses ARPACK when k << N; otherwise falls back to dense eig.
    Returns eigenvalues, right eigenvectors, left eigenvectors, with bi-orthonormalization
    so that V_L^H B V_R ≈ I (useful for participation factors).
    """
    # Ensure sparse CSR/CSC
    if not sp.issparse(A): A = sp.csr_matrix(A)
    if not sp.issparse(B): B = sp.csr_matrix(B)
    N = A.shape[0]

    use_dense = (k >= N - 2)  # ARPACK can't handle k close to N

    if not use_dense:
        # ARPACK generalized (shift-invert)
        vals, vr = spla.eigs(A, M=B, k=k, sigma=sigma)  # A v = λ B v
        # Left eigenvectors: A^T w = λ B^T w
        valsL, vl = spla.eigs(A.T, M=B.T, k=k, sigma=np.conj(sigma))
        # Match left to right by nearest λ
        idxL = [np.argmin(np.abs(valsL - lam)) for lam in vals]
        vl = vl[:, idxL]
    else:
        # Dense generalized eig (gets (almost) all finite λ); then keep k nearest to sigma
        Ad = A.toarray()
        Bd = B.toarray()
        vals_all, vr = la.eig(Ad, Bd)                 # right eigenvectors
        valsL_all, vl = la.eig(Ad.T, Bd.T)            # left eigenvectors
        # pick k closest to sigma in the complex plane (finite only)
        finite = np.isfinite(vals_all) & (np.abs(vals_all) < 1e12)
        vals_all, vr = vals_all[finite], vr[:, finite]
        # match lefts by λ
        finiteL = np.isfinite(valsL_all) & (np.abs(valsL_all) < 1e12)
        valsL_all, vl = valsL_all[finiteL], vl[:, finiteL]
        # indices of k nearest to sigma
        idx = np.argsort(np.abs(vals_all - sigma))[:min(k, len(vals_all))]
        vals, vr = vals_all[idx], vr[:, idx]
        # match left by nearest λ for those indices
        idxL = [np.argmin(np.abs(valsL_all - lam)) for lam in vals]
        vl = vl[:, idxL]

    # Bi-orthonormalize w.r.t. B: V_L^H B V_R = I
    BH = B.conj().T
    G = vl.conj().T @ (BH @ vr)
    try:
        Ginv = la.inv(G)
    except la.LinAlgError:
        # mild regularization if needed
        Ginv = la.pinv(G)
    vl = vl @ Ginv

    return vals, vr, vl

# ---------------------------
# Participation factors
# ---------------------------

def participation_factors(vr, vl):
    """
    Element-wise participation factor matrix P (states x modes):
      P[i,k] = Re( vl[i,k] * vr[i,k] )
    (after bi-orthonormalization wrt B).
    """
    P = (vl.conj() * vr).real
    # normalize each mode column to max = 1 for readability
    P = P / (np.max(np.abs(P), axis=0, keepdims=True) + 1e-15)
    return P

# ---------------------------
# Tiny demo (toy swing + algebraic "glue")
# ---------------------------

def demo_toy_primal():
    """
    Toy 2nd-order electromechanical model (δ, ω) with a dummy algebraic y
    to showcase the EX-DAE pencil and complex EM-like pair.

      δ' = ω
      M ω' = -K δ - D ω

    Algebraic 'constraint': y = Cx  -> g(x,y) = y - Cx = 0 (so gy = I, gx = -C)
    """
    M, D, K = 2.0, 0.2, 1.5
    # fx for x=[δ, ω]:  [ [0, 1],
    #                     [-K/M, -D/M] ]
    fx = np.array([[0.0, 1.0],
                   [-K/M, -D/M]])
    fy = np.zeros((2, 1))              # no direct y in f
    C  = np.array([[1.0, 0.0]])        # pick δ as “measured” algebraic
    gx = -C
    gy = np.eye(1)

    A, B = build_pencil_explicit(fx, fy, gx, gy)

    # Solve near sigma ≈ 0 (electromechanical region)
    vals, vr, vl = solve_gevp(A, B, k=3, sigma=0.0)

    # Filter finite eigenvalues (ignore the infinite one from algebraic row)
    finite = np.isfinite(vals) & (np.abs(vals) < 1e6)
    lam = vals[finite]
    vrf = vr[:, finite]
    vlf = vl[:, finite]

    # State participation (for states first: δ, ω ; last row(s) are algebraics)
    P = participation_factors(vrf[:2, :], vlf[:2, :])

    print("Eigenvalues (rad/s):")
    for z in lam:
        print(f"  λ = {z.real:+.4f} {z.imag:+.4f}j  -> f = {abs(z.imag)/(2*np.pi):.3f} Hz, ζ ≈ {-z.real/abs(z):.3f}")

    print("\nParticipation factors (rows: δ, ω; cols: modes, max=1):\n", np.round(P, 3))

    return A, B, lam, vrf, vlf, P

if __name__ == "__main__":
    # Run the toy primal GEP demo
    demo_toy_primal()
