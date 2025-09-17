
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ============================================================
# PARAMETERS — Paste paper numbers here (Ghahremani & Kamwa 2011)
# ============================================================
@dataclass
class SMIBParams:
    # Machine
    H: float = 3.5          # inertia [s] (paper-specific; replace with paper value)
    D: float = 0.0          # damping [pu torque / pu speed]
    Xd: float = 1.8         # synchronous react. d [pu]
    Xq: float = 1.7         # synchronous react. q [pu]
    Xdp: float = 0.3        # transient react. d' [pu]
    Xqp: float = 0.55       # transient react. q' [pu]
    Td0p: float = 8.0       # open-circuit d' time const [s]
    Tq0p: float = 0.4       # open-circuit q' time const [s]
    omega_b: float = 2*np.pi*60  # base elec. rad/s
    # Network / PMU
    Vt_mag: float = 1.0     # terminal voltage magnitude [pu]
    Vt_ang: float = 0.0     # terminal voltage angle [rad]

# ============================================================
# MODEL (4th-order: x=[delta, omega, Edp, Eqp], z=[P, Q])
# ============================================================
def electrical_idq(x, u, p: SMIBParams):
    delta, omega, Edp, Eqp = x
    V = u["Vt_mag"]; th = u["Vt_ang"]
    vrel = th - delta
    vd = V*np.cos(vrel); vq = V*np.sin(vrel)
    iq = (vd/p.omega_b + Eqp)/p.Xqp
    id = (Edp - vq/p.omega_b)/p.Xdp
    return id, iq, vd, vq

def electrical_power(x, u, p: SMIBParams):
    id, iq, vd, vq = electrical_idq(x,u,p)
    P = vd*id + vq*iq
    Q = vq*id - vd*iq
    return np.array([P, Q])

def f_nl(x, u, p: SMIBParams):
    delta, omega, Edp, Eqp = x
    Tm = u["Tm"]; Efd = u["Efd"]
    P, Q = electrical_power(x,u,p)
    Te = P
    ddelta = p.omega_b*(omega - 1.0)
    domega = (Tm - Te - p.D*(omega - 1.0)) / (2.0*p.H)
    # Transient emf dynamics (didactic simplification):
    id, iq, vd, vq = electrical_idq(x,u,p)
    dEdp = (-(Edp) + (p.Xd - p.Xdp)*id) / p.Td0p
    dEqp = (Efd - Eqp - (p.Xq - p.Xqp)*iq) / p.Tq0p
    return np.array([ddelta, domega, dEdp, dEqp])

def rk4(f, x, u, dt, p):
    k1 = f(x,u,p)
    k2 = f(x+0.5*dt*k1, u, dt, p) if False else f(x+0.5*dt*k1, u, p)
    k3 = f(x+0.5*dt*k2, u, p)
    k4 = f(x+dt*k3, u, p)
    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

def jacobian_wrt_x(func, x, u, p, eps=1e-6):
    y0 = np.atleast_1d(func(x, u, p))
    n = x.size; m = y0.size
    J = np.zeros((m,n))
    for i in range(n):
        dx = np.zeros_like(x); dx[i]=eps
        y1 = np.atleast_1d(func(x+dx, u, p))
        J[:,i] = (y1 - y0)/eps
    return J

# ============================================================
# EKF (known Efd)
# ============================================================
def EKF_step(xk, Pk, zk, uk, p: SMIBParams, dt, Q, R):
    # Predict
    F = np.eye(4) + dt * jacobian_wrt_x(f_nl, xk, uk, p)
    x_pred = rk4(f_nl, xk, uk, dt, p)
    P_pred = F @ Pk @ F.T + Q
    # Update
    h = lambda x: electrical_power(x, uk, p)
    H = jacobian_wrt_x(lambda xx, uu, pp: electrical_power(xx, uk, p), x_pred, uk, p)
    y_pred = h(x_pred)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ npl.inv(S)
    x_upd = x_pred + K @ (zk - y_pred)
    P_upd = (np.eye(4) - K @ H) @ P_pred
    return x_upd, P_upd

# ============================================================
# EKF-UI (unknown Efd)
# ============================================================
class EKFUI:
    def __init__(self, p: SMIBParams, dt, Qx, R, Qu):
        self.p = p; self.dt = dt
        self.Qx = Qx; self.R = R; self.Qu = Qu
        self.x = np.zeros(4); self.P = np.eye(4)*1e-2
        self.u_unknown = 1.0; self.Pu = np.array([[1e-2]])

    def set_initial(self, x0, efd0, P0=None, Pu0=None):
        self.x = x0.copy()
        self.u_unknown = efd0
        if P0 is not None: self.P = P0.copy()
        if Pu0 is not None: self.Pu = np.array([[Pu0]])

    def step(self, z_meas, known):
        # Compose input with current Efd estimate
        uk = {"Tm": known["Tm"], "Efd": self.u_unknown, "Vt_mag": known["Vt_mag"], "Vt_ang": known["Vt_ang"]}
        # Prediction
        x_pred = rk4(f_nl, self.x, uk, self.dt, self.p)
        # Linearization
        Fx = jacobian_wrt_x(f_nl, self.x, uk, self.p)
        # df/dEfd:
        def f_wrt_efd(efd):
            u2 = uk.copy(); u2["Efd"] = efd
            return f_nl(self.x, u2, self.p)
        dfdE = (f_wrt_efd(self.u_unknown+1e-6) - f_wrt_efd(self.u_unknown))/1e-6
        Ax = np.eye(4) + self.dt*Fx
        Bu = self.dt*dfdE.reshape(4,1)
        P_pred = Ax @ self.P @ Ax.T + self.Qx + Bu @ self.Pu @ Bu.T
        Pu_pred = self.Pu + np.array([[self.Qu]])

        # Measurement model
        def h(x,u): return electrical_power(x, {"Tm": known["Tm"], "Efd": u, "Vt_mag": known["Vt_mag"], "Vt_ang": known["Vt_ang"]}, self.p)
        Hx = jacobian_wrt_x(lambda xx,uu,pp: h(xx, self.u_unknown), x_pred, None, self.p)
        Hu = (h(x_pred, self.u_unknown+1e-6) - h(x_pred, self.u_unknown)).reshape(-1,1)/1e-6

        S = Hx @ P_pred @ Hx.T + Hu @ Pu_pred @ Hu.T + self.R
        Kx = P_pred @ Hx.T @ npl.inv(S)
        Ku = Pu_pred @ Hu.T @ npl.inv(S)
        z_pred = h(x_pred, self.u_unknown)
        innov = z_meas - z_pred
        x_upd = x_pred + Kx @ innov
        u_upd = self.u_unknown + float(Ku @ innov)

        I = np.eye(4)
        P_upd = (I - Kx @ Hx) @ P_pred @ (I - Kx @ Hx).T + Kx @ self.R @ Kx.T
        Pu_upd = Pu_pred - Ku @ S @ Ku.T

        self.x, self.P = x_upd, P_upd
        self.u_unknown, self.Pu = u_upd, Pu_upd
        return self.x.copy(), self.u_unknown, z_pred, innov

# ============================================================
# SIMULATION (truth + filters)
# ============================================================
def simulate_truth(p: SMIBParams, dt, T, Tm_fun, Efd_fun, V_fun, x0):
    ts = np.arange(0.0, T+dt, dt)
    X = np.zeros((len(ts),4)); Z = np.zeros((len(ts),2)); U = np.zeros((len(ts),2))
    x = x0.copy()
    for k,t in enumerate(ts):
        Tm = Tm_fun(t); Efd = Efd_fun(t); Vmag,Vang = V_fun(t)
        u = {"Tm": Tm, "Efd": Efd, "Vt_mag": Vmag, "Vt_ang": Vang}
        Z[k,:] = electrical_power(x, u, p)
        X[k,:] = x; U[k,:] = [Tm,Efd]
        x = rk4(f_nl, x, u, dt, p)
    return ts, X, Z, U

def run_demo():
    p = SMIBParams()
    dt = 0.02; T = 10.0

    # Truth inputs (edit to replicate paper scenarios)
    Tm_fun  = lambda t: 0.8 + 0.1*(t>2.0)                # step in mechanical torque
    Efd_fun = lambda t: 1.20                              # constant (unknown for EKF-UI)
    V_fun   = lambda t: (1.0, 0.0)                        # PMU terminal phasor

    x0 = np.array([0.0, 1.0, 1.0, 1.0])
    ts, Xtrue, Ztrue, Utrue = simulate_truth(p, dt, T, Tm_fun, Efd_fun, V_fun, x0)

    # PMU noise/bias (paper uses PMU-level noise; tune R accordingly)
    rng = np.random.default_rng(0)
    Zmeas = Ztrue + rng.normal(0, [0.01, 0.01], size=Ztrue.shape)

    # EKF (known Efd)
    Qx = np.diag([1e-6, 1e-5, 1e-4, 1e-4])
    Rm = np.diag([1e-4, 1e-4])
    xk = np.array([0.0, 1.02, 0.9, 1.1]); Pk = np.eye(4)*1e-2
    Xekf = np.zeros_like(Xtrue)
    for k,t in enumerate(ts):
        uk = {"Tm": Tm_fun(t), "Efd": Efd_fun(t), "Vt_mag": V_fun(t)[0], "Vt_ang": V_fun(t)[1]}
        xk, Pk = EKF_step(xk, Pk, Zmeas[k,:], uk, p, dt, Qx, Rm)
        Xekf[k,:] = xk

    # EKF-UI (estimate Efd as unknown input)
    ekfui = EKFUI(p, dt, Qx, Rm, Qu=1e-5)
    ekfui.set_initial(x0=np.array([0.0, 1.02, 0.9, 1.1]), efd0=1.0, P0=np.eye(4)*1e-2, Pu0=1e-2)
    Xui = np.zeros_like(Xtrue); Uest = np.zeros(len(ts))
    for k,t in enumerate(ts):
        known = {"Tm": Tm_fun(t), "Vt_mag": V_fun(t)[0], "Vt_ang": V_fun(t)[1]}
        xhat, uhat, zpred, innov = ekfui.step(Zmeas[k,:], known)
        Xui[k,:] = xhat; Uest[k] = uhat

    # Plots
    fig, axs = plt.subplots(3,1, figsize=(10,8), sharex=True)
    axs[0].plot(ts, Xtrue[:,0], label="δ true")
    axs[0].plot(ts, Xekf[:,0], '--', label="δ EKF")
    axs[0].plot(ts, Xui[:,0], ':', label="δ EKF-UI")
    axs[0].set_ylabel("delta [rad]"); axs[0].legend(); axs[0].grid(True)

    axs[1].plot(ts, Xtrue[:,1], label="ω true")
    axs[1].plot(ts, Xekf[:,1], '--', label="ω EKF")
    axs[1].plot(ts, Xui[:,1], ':', label="ω EKF-UI")
    axs[1].set_ylabel("omega [pu]"); axs[1].legend(); axs[1].grid(True)

    axs[2].plot(ts, Utrue[:,1], label="Efd true")
    axs[2].plot(ts, Uest, ':', label="Efd EKF-UI")
    axs[2].set_ylabel("Efd [pu]"); axs[2].set_xlabel("t [s]"); axs[2].legend(); axs[2].grid(True)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    run_demo()
