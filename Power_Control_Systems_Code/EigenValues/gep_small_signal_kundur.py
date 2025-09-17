# gep_small_signal_kundur.py
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg as la

def participation_factors(vr, vl):
    """Bi-orthonormalize (B=I) then elementwise real part; column-normalize to max=1."""
    G = vl.conj().T @ vr
    try:
        vl = vl @ la.inv(G)
    except la.LinAlgError:
        vl = vl @ la.pinv(G)
    P = (vl.conj() * vr).real
    return P / (np.max(np.abs(P), axis=0, keepdims=True) + 1e-15)

def solve_modes(A, k=8, sigma=0.0):
    """Dense fallback if ARPACK guardrail triggers."""
    n = A.shape[0]
    use_dense = (k >= n - 2)
    if use_dense:
        vals, vr = la.eig(A)
        # pick k closest to sigma
        idx = np.argsort(np.abs(vals - sigma))[:min(k, len(vals))]
        vals, vr = vals[idx], vr[:, idx]
        # left eigenvectors from A^T
        valsL, vl_all = la.eig(A.T)
        idxL = [np.argmin(np.abs(valsL - lam)) for lam in vals]
        vl = vl_all[:, idxL]
    else:
        vals, vr = spla.eigs(A, k=k, sigma=sigma)
        valsL, vl = spla.eigs(A.T, k=k, sigma=np.conj(sigma))
        idxL = [np.argmin(np.abs(valsL - lam)) for lam in vals]
        vl = vl[:, idxL]
    return vals, vr, vl

def build_kundur_2area():
    """
    Classical 4-machine, 2-area model (lossless, classical machines).
    States: x = [δ1..δ4, ω1..ω4]^T
    δ' = ω
    M ω' = -K δ - D ω
    """
    # ---- TUNED for low inter-area frequency ----
    H = np.array([25.0, 25.0, 25.0, 25.0])   # larger inertia → lower freq
    w_s = 2*np.pi*60.0
    M = 2.0*H / w_s                          # pu·s^2 “mass”
    D = np.array([0.5, 0.5, 0.5, 0.5])       # small positive damping

    # Susceptance Laplacian
    Bij = np.zeros((4,4))
    def add_line(i,j,B):
        Bij[i,j] += B; Bij[j,i] += B
        Bij[i,i] -= B; Bij[j,j] -= B

    # Strong inside each area (kept sizeable so local modes remain high)
    add_line(0,1, 8.0)     # area 1 internal
    add_line(2,3, 8.0)     # area 2 internal
    # Very weak tie → inter-area drops toward ~0.5–0.7 Hz
    add_line(1,2, 0.05)

    # Synchronizing (stiffness) matrix K: with E≈1, δ≈0 → K = -Bij
    K = -Bij.copy()
    Minv = np.diag(1.0/M)
    Dmat = np.diag(D)

    Z = np.zeros((4,4)); I = np.eye(4)
    A = np.block([[Z,                 I],
                  [-Minv @ K, -Minv @ Dmat]])
    return A, K, M, D

def demo_kundur_2area():
    A, K, M, D = build_kundur_2area()
    vals, vr, vl = solve_modes(A, k=8, sigma=0.0)

    # Sort by damping ratio (least damped first)
    alpha, beta = vals.real, vals.imag
    freq = np.abs(beta) / (2*np.pi)
    zeta = -alpha / np.maximum(np.sqrt(alpha**2 + beta**2), 1e-15)
    order = np.argsort(zeta)
    vals, vr, vl, freq, zeta = vals[order], vr[:,order], vl[:,order], freq[order], zeta[order]

    # Participation (angles only for readability)
    P = participation_factors(vr, vl)
    P_delta = P[:4, :]  # rows δ1..δ4

    print("Electromechanical-like modes (top few):")
    for k in range(min(8, len(vals))):
        lam = vals[k]
        print(f"  λ = {lam.real:+.4f} {lam.imag:+.4f}j   f={freq[k]:.3f} Hz, ζ={zeta[k]:.3f}")

    # ---- Inter-area picker: focus on 0.3–1.2 Hz & ζ≥0; require opposite-area motion
    def interarea_score(vr_angle):
        # phase-align to δ1, measure opposite means between (δ1,δ2) vs (δ3,δ4)
        phase = np.angle(vr_angle[0])
        d = (vr_angle * np.exp(-1j*phase)).real
        s12 = np.mean(d[0:2]); s34 = np.mean(d[2:4])
        # penalty if same sign; reward opposition; weight by similarity within areas
        opposition = -abs(s12 + s34)
        coherence = - (abs(d[0]-d[1]) + abs(d[2]-d[3]))
        return opposition + 0.1*coherence, s12, s34, d

    band = (freq >= 0.3) & (freq <= 1.2) & (zeta >= 0)
    cand = np.where(band)[0]
    if cand.size == 0:
        # expand band if nothing found (shouldn't happen with tuned params)
        cand = np.where((freq >= 0.1) & (freq <= 2.0) & (zeta >= 0))[0]

    best_score = -np.inf
    best_k = None
    for k in cand:
        s, s12, s34, d = interarea_score(vr[:4, k])
        if s > best_score:
            best_score, best_k, best_s, best_d = s, k, (s12, s34), d

    print("\nAngle participations (max=1) for the least-damped few:")
    names = ["δ1","δ2","δ3","δ4"]
    for k in range(min(4, P_delta.shape[1])):
        print(f"  mode {k}: " + ", ".join(f"{n}={P_delta[i,k]:.2f}" for i,n in enumerate(names)))

    print(f"\nLikely inter-area mode ≈ mode {best_k}: f={freq[best_k]:.3f} Hz, ζ={zeta[best_k]:.3f}")
    print("  Area means: A1(δ1,δ2)≈{:.3f}, A2(δ3,δ4)≈{:.3f}".format(best_s[0], best_s[1]))
    print("  Right-eig. angle components (phase-aligned):",
          " ".join(f"{names[i]}={best_d[i]:+.3f}" for i in range(4)))

if __name__ == "__main__":
    demo_kundur_2area()
