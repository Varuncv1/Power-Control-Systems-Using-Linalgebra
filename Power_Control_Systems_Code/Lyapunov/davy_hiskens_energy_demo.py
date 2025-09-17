import numpy as np
import matplotlib.pyplot as plt

# ===========================
# Parameters (tweak freely)
# ===========================
class P:
    # Lossless susceptance matrix B (3-bus: 1=G1 internal (infinite-ish), 2=G2 internal, 3=load bus)
    # Lines: x12=0.20, x13=0.25, x23=0.40  =>  y = 1/(j x) = -j/x  =>  B = imag(Y)
    x12, x13, x23 = 0.20, 0.25, 0.40
    y12, y13, y23 = -1/0.20, -1/0.25, -1/0.40  # imag parts only
    B = np.array([
        [-(y12+y13),   y12,          y13       ],
        [   y12,     -(y12+y23),     y23       ],
        [   y13,        y23,       -(y13+y23)  ],
    ], dtype=float)  # this is imag(Y), i.e., susceptance (no shunts)

    # Generator 2 (classical machine) inertia/damping/mech power
    M2   = 0.2       # pu*s^2
    D2   = 0.0       # pu*s  (COA derivation often sets D=0; we keep 0 here)
    Pm2  = 0.9       # pu

    # Internal emf magnitudes (treated fixed for classical model)
    E1   = 1.05
    E2   = 1.05

    # Dynamic load at bus 3
    P_load = 1.00    # pu  (constant real power component)
    Q0     = 0.40    # pu  (steady-state reactive baseline)
    alpha  = 0.10    # Q_tr(V) = alpha ln V + beta
    beta   = 0.20
    Tq     = 1.0     # s (load reactive recovery time)

    # Algebraic “power-flow correction” gains for load bus (gradient flow solver)
    # These emulate the singular-perturbation epsilons in the paper; small → stiff
    eps_P = 0.05
    eps_Q = 0.05

    # Fault (line opening) scenario
    t_end        = 12.0
    dt           = 0.002
    disturb_time = 2.0      # open line (1,2)
    restore_time = 3.5      # reclose line (1,2)
    toggle_pair  = (1, 2)   # 0-based indices of buses to perturb connectivity
    toggle_deltaB= +3.0     # temporarily ADD susceptance |B| between pair (net effect like topology change)

    # “Critical” energy for demo (paper’s 2-machine example reports ~0.0404 pu at UEP)
    Ecrit = 0.0404

# ---------------------------
# Network helper
# ---------------------------
def apply_line_toggle(B, t):
    """Mimic a line opening/reclosing by modifying the susceptance between a bus pair during a time window."""
    i, j = P.toggle_pair
    Bmod = B.copy()
    if P.disturb_time <= t < P.restore_time:
        # Remove coupling by making the mutual term smaller in magnitude.
        # Here: increase diagonal magnitudes and decrease mutual in a balanced way.
        Bmod[i, j] -= P.toggle_deltaB
        Bmod[j, i] -= P.toggle_deltaB
        Bmod[i, i] += P.toggle_deltaB
        Bmod[j, j] += P.toggle_deltaB
    return Bmod

def electrical_PQ(bus_angles, Vmag, B):
    """
    Power through a lossless network with susceptance matrix B = imag(Y).
    Using standard structure-preserving sin/cos forms for P, Q at each bus.
    """
    n = len(Vmag)
    Pnet = np.zeros(n)
    Qnet = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                # For pure susceptance B, the self term contributes -B_ii*V_i^2 to Q (network charging)
                Qnet[i] += -B[i, i]*Vmag[i]**2
            Vij = Vmag[i]*Vmag[j]
            aij = bus_angles[i] - bus_angles[j]
            Pnet[i] += Vij * B[i, j] * np.sin(aij)   # G=0 → P ~ B*sin
            Qnet[i] += Vij * B[i, j] * np.cos(aij)   # Q ~ -B_ii*Vi^2 + sum B_ij ViVj cos
    return Pnet, Qnet

# ---------------------------
# Energy (Lyapunov) function
# ---------------------------
def lyapunov_energy(delta2, omega2, V3, t, xi):
    """
    Paper-consistent energy pieces (cf. eqs. (33)/(52)):
      E_kin = 0.5 M ω^2
      E_net = sum_{i<j} V_i V_j B_ij (1 - cos(θ_i - θ_j))
      E_load = 0.5 α (ln V)^2 + β ln V - ξ ln V
    where ξ is the dynamic load state (Q_load = Q0 + ξ), and V is the load-bus magnitude.
    """
    B = apply_line_toggle(P.B, t)
    theta = np.array([0.0, delta2, 0.0])              # bus 1 angle is reference; bus 3 angle set as 0 for simplicity (algebraic θ3 below)
    Vmag  = np.array([P.E1, P.E2, V3])

    # Kinetic energy (generator 2)
    Ek = 0.5 * P.M2 * omega2**2

    # Network "potential" energy (pairwise)
    Ep = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            Ep += Vmag[i]*Vmag[j]*B[i, j]*(1.0 - np.cos(theta[i] - theta[j]))

    # Load energy term (logarithmic transient + ξ coupling)
    Vclip = max(V3, 1e-6)
    El = 0.5*P.alpha*(np.log(Vclip))**2 + P.beta*np.log(Vclip) - xi*np.log(Vclip)
    return Ek + Ep + El

# ---------------------------
# ODE/DAE (structure-preserving with algebraic correction at load bus)
# ---------------------------
def f_state(t, x):
    """
    States: x = [delta2, omega2, V3, xi, theta3]
      - Generator 2 swing: d(delta2)=omega2; d(omega2) = (Pm2 - Pe2 - D2*omega2)/M2
      - Dynamic load:  dxi = (-xi + Q_tr(V3))/Tq,  Q_tr = α ln V + β
      - Load bus algebraic correction steps:  d(theta3) ∝ -P_mismatch,  d(V3) ∝ -Q_mismatch
    """
    delta2, omega2, V3, xi, theta3 = x
    B = apply_line_toggle(P.B, t)
    theta = np.array([0.0, delta2, theta3])
    Vmag  = np.array([P.E1, P.E2, max(V3, 1e-6)])

    Pnet, Qnet = electrical_PQ(theta, Vmag, B)

    # Generator electrical air-gap power at bus 2
    Pe2 = Pnet[1]
    domega2 = (P.Pm2 - Pe2 - P.D2*omega2) / P.M2
    ddelta2 = omega2

    # Dynamic load recovery
    Vclip = max(V3, 1e-6)
    Qtr = P.alpha*np.log(Vclip) + P.beta
    dxi = (-xi + Qtr) / P.Tq
    Qload = P.Q0 + xi
    Pload = P.P_load

    # Load-bus power mismatches (sign convention: injections positive)
    Pmis = Pnet[2] + Pload
    Qmis = Qnet[2] + Qload

    # Algebraic correction dynamics for load-bus voltage and angle
    dtheta3 = - Pmis / max(P.eps_P, 1e-6)
    dV3     = - Qmis / max(P.eps_Q, 1e-6)

    return np.array([ddelta2, domega2, dV3, dxi, dtheta3])

def rk4_step(func, t, x, h):
    k1 = func(t, x)
    k2 = func(t + 0.5*h, x + 0.5*h*k1)
    k3 = func(t + 0.5*h, x + 0.5*h*k2)
    k4 = func(t + h, x + h*k3)
    return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ---------------------------
# Main simulation
# ---------------------------
def run_sim():
    tgrid = np.arange(0.0, P.t_end + P.dt, P.dt)
    x = np.zeros((len(tgrid), 5))
    # Initial conditions near SEP (flat-ish)
    x[0] = np.array([0.0, 0.0, 1.02, 0.0, 0.0])  # [δ2, ω2, V3, ξ, θ3]

    E = np.zeros_like(tgrid)
    first_cross = None

    for k in range(len(tgrid)-1):
        t = tgrid[k]
        # energy BEFORE stepping, to track crossing
        E[k] = lyapunov_energy(x[k,0], x[k,1], x[k,2], t, x[k,3])

        # detect first time E exceeds chosen critical energy
        if first_cross is None and E[k] >= P.Ecrit:
            first_cross = t

        # advance
        x[k+1] = rk4_step(f_state, t, x[k], P.dt)

        # guard rails for numerical safety
        x[k+1,2] = np.clip(x[k+1,2], 0.3, 1.8)             # V3 in [0.3, 1.8]
        x[k+1,0] = ( (x[k+1,0] + np.pi) % (2*np.pi) ) - np.pi  # wrap δ2
        x[k+1,4] = ( (x[k+1,4] + np.pi) % (2*np.pi) ) - np.pi  # wrap θ3

    # last energy
    E[-1] = lyapunov_energy(x[-1,0], x[-1,1], x[-1,2], tgrid[-1], x[-1,3])
    return tgrid, x, E, first_cross

def main():
    t, X, E, tcrit = run_sim()
    δ2, ω2, V3, ξ, θ3 = X.T

    print(f"First energy crossing of Ecrit={P.Ecrit:.4f} pu:", 
          ("None (never crossed)" if tcrit is None else f"t = {tcrit:.3f} s"))

    fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    axs[0].plot(t, δ2); axs[0].set_ylabel("δ₂ (rad)"); axs[0].grid(ls='--', alpha=0.5)
    axs[1].plot(t, ω2); axs[1].set_ylabel("ω₂ (pu)");  axs[1].grid(ls='--', alpha=0.5)
    axs[2].plot(t, V3, label="V₃"); axs[2].plot(t, ξ+P.Q0, label="Q₃ (≈Q0+ξ)")
    axs[2].legend(); axs[2].set_ylabel("V₃, Q₃ (pu)"); axs[2].grid(ls='--', alpha=0.5)
    axs[3].plot(t, E, label="E(t)")
    axs[3].axhline(P.Ecrit, color='k', linestyle=':', label="Ecrit")
    axs[3].axvspan(P.disturb_time, P.restore_time, color='orange', alpha=0.15, label="Line open")
    axs[3].set_ylabel("Lyapunov-like E (pu)"); axs[3].set_xlabel("Time (s)")
    axs[3].legend(); axs[3].grid(ls='--', alpha=0.5)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
