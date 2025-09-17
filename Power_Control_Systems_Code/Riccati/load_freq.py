# Create a complete, runnable Python script that:
# - builds the linearized multi-area LFC model exactly in the paper's state order
# - assembles A,B,E,F,C per area and couples them (Eq. 10–11)
# - solves CARE for LQR (Sec. 3.1.2) with user-set Q,R (paper provides a sample)
# - implements a lower-layer discrete PI per area acting on ACE (Eq. 40)
# - enforces GRC and simple actuator saturation (Eqs. 35–36) and |Δf| limit (Eq. 37)
# - simulates step and random net disturbances
# - reports settling time and overshoot for Δf and P_tie
# - optional (toggle) steady-state KF (structure only; uses simple LQE from CARE)
#
# NOTE: Fill numeric parameters from Table 2 by editing 'default_params()' below.
#       Values for M_i and W0_ij are not in the snippet; set your system's values.
#
# Dependencies: numpy, scipy, matplotlib
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

@dataclass
class AreaParams:
    M: float            # inertia constant Mi
    D: float            # damping Di
    R: float            # droop Ri
    TT: float           # turbine time constant TTi
    TGov: float         # governor time constant TGovi
    beta: float         # frequency bias beta_i
    f_lim: float        # |Δf_i| limit (Hz)
    # PI gains (lower level). You can tune/override these.
    Kp: float
    Ki: float

@dataclass
class TieParams:
    W0: Dict[Tuple[int,int], float]  # synchronizing coefficients W0_ij (i<->j)

@dataclass
class SimConfig:
    Ts: float = 0.1              # sampling time for PI update [s]
    umax: float = 0.015          # actuator saturation on P_sc per unit (per paper scenarios)
    grc: float = 1.7e-3          # GRC limit in pu/s (Eq. 36)
    t_final: float = 30.0        # sim duration [s]
    use_kf: bool = False         # optional LQE/KF estimation (structure only)

def default_params() -> Tuple[Dict[int, AreaParams], TieParams]:
    # Fill with Table 2 where available; choose reasonable placeholders for missing Mi and W0_ij.
    # Paper's Table 2 snippets (units as shown):
    # beta1=0.43, beta2=0.41, beta3=0.38; D1=2.25, D2=2.42, D3=2.28 (puMW/Hz)
    # R1=1.95, R2=1.84, R3=1.98; TT1=0.32, TT2=0.30, TT3=0.31 (s); TGov1=0.15, TGov2=0.12, TGov3=0.16 (s)
    # Ts=0.1 s; f_lim=0.2 Hz; GRC ±0.1 pu/min ≈ 1.7e-3 pu/s.
    # Missing: Mi (choose 10.0 s as placeholder), synchronizing W0_ij (choose 2π*0.05 as placeholder)
    areas = {
        1: AreaParams(M=10.0, D=2.25, R=1.95, TT=0.32, TGov=0.15, beta=0.43, f_lim=0.2, Kp=0.34, Ki=0.124),
        2: AreaParams(M=10.0, D=2.42, R=1.84, TT=0.30, TGov=0.12, beta=0.41, f_lim=0.2, Kp=0.263, Ki=0.221),
        3: AreaParams(M=10.0, D=2.28, R=1.98, TT=0.31, TGov=0.16, beta=0.38, f_lim=0.2, Kp=0.345, Ki=0.255),
    }
    # symmetric ties; W0_ij in pu/MW-s (placeholder coherence factor)
    w = 0.05
    W0 = {(1,2): w, (2,1): w, (1,3): w, (3,1): w, (2,3): w, (3,2): w}
    return areas, TieParams(W0=W0)

def area_matrices(i: int, areas: Dict[int, AreaParams], ties: TieParams, neighbors: List[int]) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Single-area matrices (Eq. 10–11). State x_i = [Δf_i, P_G_i, P_gov_i, P_tie_i]^T
    xdot = A_i x_i + B_i u_i + E_i P_net_i + F_i * sum_j(Δf_j)  with tie coupling via W0_ij and Δf.
    y_i = C_i x_i  with ACE_i = [beta_i, 0, 0, 1] x_i
    """
    p = areas[i]
    A = np.zeros((4,4))
    # row for Δf_i
    A[0,0] = -p.D/p.M
    A[0,1] =  1.0/p.M
    A[0,3] = -1.0/p.M
    # row for P_G_i
    A[1,1] = -1.0/p.TT
    A[1,2] =  1.0/p.TT
    # row for P_gov_i
    A[2,0] = -1.0/(p.R*p.TGov)
    A[2,2] = -1.0/p.TGov
    # row for P_tie_i
    # sum of 2π W0_ij multiplies (Δf_i - Δf_j) enters P_tie_i dynamics:
    sumW = sum(2*np.pi*ties.W0[(i,j)] for j in neighbors)
    A[3,0] =  sumW
    # A[3,?] has no dependence on P_G or P_gov
    # B_i for u_i = P_sc,i
    B = np.zeros((4,1)); B[2,0] = 1.0/p.TGov
    # E_i for local net disturbance P_net,i affects Δf_i only
    E = np.zeros((4,1)); E[0,0] = -1.0/p.M
    # F_i couples other areas' Δf_j into P_tie_i row negatively with coeff -2π W0_ij
    F = np.zeros((4, len(neighbors)))
    for idx, j in enumerate(neighbors):
        F[3, idx] = -2*np.pi*ties.W0[(i,j)]
    # C_i for ACE: [beta_i, 0, 0, 1]
    C = np.array([[p.beta, 0.0, 0.0, 1.0]])
    return A, B, E, F, C

def assemble_global(areas: Dict[int, AreaParams], ties: TieParams) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, Dict[int,Tuple[int,List[int]]]]:
    ids = sorted(areas.keys())
    nA = len(ids)
    # layout: [x1; x2; x3] with each 4 states
    n = 4*nA
    A = np.zeros((n,n)); B = np.zeros((n,nA)); E = np.zeros((n,nA)); C = np.zeros((nA,n))
    # mapping for neighbor indices
    index_map = {}
    for k,i in enumerate(ids):
        neighbors = [j for j in ids if j!=i and (i,j) in ties.W0]
        index_map[i] = (k, neighbors)
        Ai,Bi,Ei,Fi,Ci = area_matrices(i, areas, ties, neighbors)
        # place Ai
        A[k*4:(k+1)*4, k*4:(k+1)*4] = Ai
        # place B column for u_i
        B[k*4:(k+1)*4, k] = Bi[:,0]
        # place E column for local disturbance d_i
        E[k*4:(k+1)*4, k] = Ei[:,0]
        # couple neighbors via Fi on their Δf_j -> x_j[0]
        for idx,j in enumerate(neighbors):
            jpos = ids.index(j)
            # Fi affects row block (P_tie_i row = 3) at col x_j Δf (index 0 inside block)
            A[k*4+3, jpos*4+0] += Fi[3, idx]
        # C for ACE_i
        C[k, k*4:(k+1)*4] = Ci[0,:]
    return A, B, E, C, np.eye(nA), index_map

def lqr_gain(A: NDArray, B: NDArray, Q: NDArray, R: NDArray) -> NDArray:
    """Solve continuous-time LQR via CARE: K = R^{-1} B^T P"""
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, (B.T @ P))
    return K

def simulate(system: Tuple[NDArray,NDArray,NDArray,NDArray,NDArray,Dict], areas: Dict[int,AreaParams],
             cfg: SimConfig, d_profile, case_name="step"):
    A,B,E,C,_,index_map = system
    ids = sorted(areas.keys())
    nA = len(ids)
    n = 4*nA
    # Upper-layer LQR (uses all states)
    # Paper gives a sample Q,R (Sec. 3.1.2): Q = diag(9.6, 1.4, 1.2, 9.6) per area; R = diag(1.1,0.9,0.9,1.1) ???
    # We'll build block-diag Q across areas; for R, we need nA-by-nA (each u_i): use diag([1.0]*nA) as baseline.
    Qi = np.diag([9.6, 1.4, 1.2, 9.6])
    Q = np.kron(np.eye(nA), Qi)
    R = np.diag([1.0]*nA)
    K = lqr_gain(A, B, Q, R)  # shape (nA, n)
    # Lower-layer PI (discrete) acting on ACE per area
    # initialize states and integrators
    x0 = np.zeros(n)
    ace_int = np.zeros(nA)
    # Helper to extract ACE
    def ACE(x):
        return (C @ x)
    # Disturbance function d(t) per-area column vector length nA
    def d_of_t(t):
        return d_profile(t)
    # Saturation and GRC tracking for each area's u (P_sc)
    u_prev = np.zeros(nA)
    t_prev = 0.0
    # ODE of plant
    def f(t, x):
        nonlocal u_prev, t_prev, ace_int
        # Update PI once per Ts (zero-order hold)
        if t - t_prev >= cfg.Ts or np.isclose(t, 0.0):
            ace = ACE(x).reshape(-1)
            # clamp |Δf| inside f_lim by penalizing ACE (soft)
            for k,i in enumerate(ids):
                if abs(x[k*4+0]) > areas[i].f_lim:
                    ace[k] = np.sign(ace[k])*0.5*abs(ace[k])
            # Discrete PI update (parallel form)
            for k,i in enumerate(ids):
                Kp, Ki = areas[i].Kp, areas[i].Ki
                ace_int[k] += ace[k]*cfg.Ts
                u_pi = Kp*ace[k] + Ki*ace_int[k]
                # Upper-layer LQR sets reference offset: u_ref = -K x
                u_ref = - (K @ x)[k]
                u_cmd = u_ref + u_pi
                # Apply GRC and saturation
                du = np.clip(u_cmd - u_prev[k], -cfg.grc*cfg.Ts, cfg.grc*cfg.Ts)
                u_prev[k] = np.clip(u_prev[k] + du, -cfg.umax, cfg.umax)
            t_prev = t
        # Plant dynamics
        dx = (A @ x) + (B @ u_prev) + (E @ d_of_t(t))
        return dx
    t_eval = np.linspace(0.0, cfg.t_final, int(cfg.t_final/0.01)+1)
    sol = solve_ivp(f, [0.0, cfg.t_final], x0, t_eval=t_eval, rtol=1e-7, atol=1e-9, max_step=0.02)
    t = sol.t; X = sol.y.T  # shape (T, n)
    # Metrics
    def overshoot(sig):
        m = np.max(sig)
        return m if m>0 else 0.0
    def undershoot(sig):
        m = np.min(sig)
        return m if m<0 else 0.0
    def settling_time(sig, tol=0.02):
        final = sig[-1]
        band = tol*max(1.0, abs(final))  # 2% of 1 pu
        for idx in range(len(sig)-1, -1, -1):
            if abs(sig[idx]-final)>band:
                return t[idx+1] if idx+1<len(t) else t[-1]
        return 0.0
    # Collect Δf and P_tie
    metrics = {}
    for k,i in enumerate(ids):
        df = X[:, k*4+0]
        pt = X[:, k*4+3]
        metrics[f"area{i}_df_overshoot"] = overshoot(df)
        metrics[f"area{i}_df_undershoot"] = undershoot(df)
        metrics[f"area{i}_df_ts"] = settling_time(df)
        metrics[f"area{i}_pt_overshoot"] = overshoot(pt)
        metrics[f"area{i}_pt_undershoot"] = undershoot(pt)
        metrics[f"area{i}_pt_ts"] = settling_time(pt)
    return t, X, metrics

def step_disturbance(area: int, mag: float, t0: float):
    def d(t):
        return np.array([mag if (a==area and t>=t0) else 0.0 for a in [1,2,3]])
    return d

def random_disturbance(seed=0):
    rng = np.random.default_rng(seed)
    # simple band-limited random net disturbance in CA1 and CA3
    # use a low-pass filtered white noise
    t_cache = []; d_cache = []
    def d(t):
        if t_cache and np.isclose(t, t_cache[-1]):
            return d_cache[-1]
        # update every 0.1 s
        if not t_cache or t - t_cache[-1] >= 0.1:
            # OU-ish increment
            prev = d_cache[-1] if d_cache else np.zeros(3)
            mu = np.array([0.0, 0.0, 0.0])
            theta = 2.0
            sigma = 0.02
            dt = 0.1
            x = prev + theta*(mu - prev)*dt + sigma*np.sqrt(dt)*rng.standard_normal(3)
            x[1] = 0.0  # keep CA2 steady
            d_cache.append(x); t_cache.append(t)
        return d_cache[-1]
    return d

def plot_results(t, X, title=""):
    plt.figure()
    for k,i in enumerate([1,2,3]):
        plt.plot(t, X[:, k*4+0], label=f"Δf{i} [Hz]")
    plt.title(title + " – Frequency deviations")
    plt.xlabel("t [s]"); plt.ylabel("Δf [Hz]")
    plt.legend(); plt.grid(True)
    plt.figure()
    for k,i in enumerate([1,2,3]):
        plt.plot(t, X[:, k*4+3], label=f"P_tie{i} [pu]")
    plt.title(title + " – Tie-line power deviations")
    plt.xlabel("t [s]"); plt.ylabel("P_tie [pu]")
    plt.legend(); plt.grid(True)

if __name__ == "__main__":
    areas, ties = default_params()
    system = assemble_global(areas, ties)
    cfg = SimConfig(Ts=0.1, t_final=25.0)
    # CASE 1: Step net disturbance in CA1 of 0.1 pu at t=5 s (as in paper's scenario)
    d1 = step_disturbance(area=1, mag=0.1, t0=5.0)
    t, X, M = simulate(system, areas, cfg, d1, case_name="step")
    print("=== Step case metrics ===")
    for k,v in M.items():
        print(f"{k}: {v:.4f}")
    plot_results(t, X, "Step (0.1 pu in CA1 @ 5 s)")
    # CASE 2: Random net disturbances in CA1 & CA3
    d2 = random_disturbance(seed=1)
    t2, X2, M2 = simulate(system, areas, cfg, d2, case_name="random")
    print("\n=== Random case metrics ===")
    for k,v in M2.items():
        print(f"{k}: {v:.4f}")
    plot_results(t2, X2, "Random net disturbances (CA1 & CA3)")
    plt.show()
