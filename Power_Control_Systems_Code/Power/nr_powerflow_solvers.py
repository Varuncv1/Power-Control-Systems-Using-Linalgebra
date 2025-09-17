import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import sparseqr  # SuiteSparseQR python wrapper (pip install sparseqr)
    HAS_SPARSEQR = True
except Exception:
    HAS_SPARSEQR = False


class NRSolver:
    """
    Newton–Raphson power flow with pluggable linear solver: LU (SuperLU) or QR (SuiteSparseQR).

    Inputs you must provide:
      - Ybus: complex CSR matrix (N x N)
      - bus_type: int array of length N with {0: Slack, 1: PV, 2: PQ}
      - P_spec, Q_spec: specified injections (per-unit) for each bus (N,)
      - V0, th0: initial guesses (N,) for magnitudes and angles [rad]

    Notes:
      - Matches the thesis workflow (mismatch -> Jacobian -> solve -> update) and uses CSR everywhere.
      - Jacobian diagonals use the same simplifications from Ch.3 to reduce ops:
          J1_ii = -Q_i - V_i^2 * B_ii
          J2_ii =  P_i / V_i + V_i * G_ii
          J3_ii =  P_i - V_i^2 * G_ii
          J4_ii =  Q_i / V_i - V_i * B_ii
        (derived from the power equations)  # see Chapter 3 listings
    """

    SLACK, PV, PQ = 0, 1, 2

    def __init__(self, Ybus_csr, bus_type, P_spec, Q_spec, V0, th0,
                 tol=1e-8, max_it=20, reordering='COLAMD', solver='lu'):
        assert sp.isspmatrix_csr(Ybus_csr)
        self.Y = Ybus_csr
        self.N = Ybus_csr.shape[0]
        self.bus_type = np.asarray(bus_type)
        self.V = np.asarray(V0, dtype=float).copy()
        self.th = np.asarray(th0, dtype=float).copy()
        self.P_spec = np.asarray(P_spec, dtype=float)
        self.Q_spec = np.asarray(Q_spec, dtype=float)
        self.tol = tol
        self.max_it = max_it
        self.solver = solver.lower()
        self.permc = {'COLAMD':'COLAMD','AMD':'MMD_AT_PLUS_A','NATURAL':'NATURAL'}.get(reordering.upper(),'COLAMD')

        # bus partitions
        self.pv = np.where(self.bus_type == self.PV)[0]
        self.pq = np.where(self.bus_type == self.PQ)[0]
        self.slack = np.where(self.bus_type == self.SLACK)[0]
        assert self.slack.size == 1, "Exactly one slack bus is required"
        self.slack_idx = int(self.slack[0])

        # variable ordering: [angles for non-slack buses; volt mags for PQ buses]
        self.ang_idx = np.array([i for i in range(self.N) if i != self.slack_idx], dtype=int)
        self.V_idx = self.pq.copy()

    # ----- core power equations -----
    def power_injections(self):
        # I = Y*V (phasors); P + jQ = V * conj(I)
        Vc = self.V * np.exp(1j * self.th)
        I = self.Y.dot(Vc)
        S = Vc * np.conj(I)
        P = S.real
        Q = S.imag  # sign convention (power injected)
        return P, Q

    def mismatch(self):
        P, Q = self.power_injections()
        dP = self.P_spec - P
        dQ = self.Q_spec - Q
        # build stacked mismatch for variables we actually solve:
        # angles: all except slack; volt mags: PQ only
        mis = np.concatenate([dP[self.ang_idx], dQ[self.pq]])
        return mis, P, Q

    # ----- Jacobian assembly (CSR), matching Ch.3 structure -----
    def jacobian(self, P, Q):
        G = self.Y.real
        B = self.Y.imag

        # helpful local copies
        V = self.V
        th = self.th

        # we’ll build with triplets then CSR
        rows = []
        cols = []
        vals = []

        # Convenience: map "reduced" row/col positions
        # angle block rows/cols (excluding slack)
        ang_pos = {bus:i for i,bus in enumerate(self.ang_idx)}
        # voltage block rows/cols (PQ only)
        Vpos = {bus:i for i,bus in enumerate(self.pq)}

        # iterate Y nonzeros (CSR)
        Y = self.Y.tocsr()
        indptr, indices, data = Y.indptr, Y.indices, Y.data

        # Off-diagonal terms (depend on sin/cos(theta_i - theta_j))
        for i in range(self.N):
            for k in range(indptr[i], indptr[i+1]):
                j = indices[k]
                Yij = data[k]
                Gij, Bij = Yij.real, Yij.imag
                if i == j:
                    continue
                dth = th[i] - th[j]
                ViVj = V[i]*V[j]
                # contributions to J1..J4 for off-diagonals
                # J1(i,j) = Vi*Vj*( Gij*sin(dth) - Bij*cos(dth) )
                if i != self.slack_idx and j != self.slack_idx:
                    rows.append(ang_pos[i]); cols.append(ang_pos[j])
                    vals.append(ViVj*( Gij*np.sin(dth) - Bij*np.cos(dth)))

                # J2(i,j) = V_i * ( Gij*cos(dth) + Bij*sin(dth) )
                if i != self.slack_idx and j in self.pq:
                    rows.append(ang_pos[i]); cols.append(len(self.ang_idx)+Vpos[j])
                    vals.append(V[i]*( Gij*np.cos(dth) + Bij*np.sin(dth)))

                # J3(i,j) = Vi*Vj*( -Gij*cos(dth) - Bij*sin(dth) )
                if i in self.pq and j != self.slack_idx:
                    rows.append(len(self.ang_idx)+Vpos[i]); cols.append(ang_pos[j])
                    vals.append(ViVj*( -Gij*np.cos(dth) - Bij*np.sin(dth)))

                # J4(i,j) = V_i * ( Gij*sin(dth) - Bij*cos(dth) )
                if i in self.pq and j in self.pq:
                    rows.append(len(self.ang_idx)+Vpos[i]); cols.append(len(self.ang_idx)+Vpos[j])
                    vals.append(V[i]*( Gij*np.sin(dth) - Bij*np.cos(dth)))

        # Diagonals via the Chapter 3 simplifications (saves work)
        # J1(ii) = -Q_i - V_i^2 * B_ii
        # J2(ii) =  P_i / V_i + V_i * G_ii
        # J3(ii) =  P_i - V_i^2 * G_ii
        # J4(ii) =  Q_i / V_i - V_i * B_ii
        Bdiag = B.diagonal()
        Gdiag = G.diagonal()
        for i in range(self.N):
            if i != self.slack_idx:
                rows.append(ang_pos[i]); cols.append(ang_pos[i])
                vals.append(-Q[i] - (V[i]**2)*Bdiag[i])  # J1(ii)

                if i in self.pq:
                    rows.append(ang_pos[i]); cols.append(len(self.ang_idx)+Vpos[i])
                    vals.append(P[i]/max(V[i],1e-12) + V[i]*Gdiag[i])  # J2(ii)

            if i in self.pq:
                rows.append(len(self.ang_idx)+Vpos[i]); cols.append(ang_pos[i])
                vals.append(P[i] - (V[i]**2)*Gdiag[i])  # J3(ii)

                rows.append(len(self.ang_idx)+Vpos[i]); cols.append(len(self.ang_idx)+Vpos[i])
                vals.append(Q[i]/max(V[i],1e-12) - V[i]*Bdiag[i])  # J4(ii)

        nvar = len(self.ang_idx) + len(self.pq)
        J = sp.csr_matrix((vals, (rows, cols)), shape=(nvar, nvar))
        return J

    # ----- solves -----
    def solve_linear(self, J, rhs):
        if self.solver == 'lu':
            # SuperLU; reordering options per Chapter 4 comparisons
            # ('COLAMD', 'MMD_AT_PLUS_A'(AMD-like), 'NATURAL')
            t0 = time.perf_counter()
            lu = spla.splu(J.tocsc(), permc_spec=self.permc, diag_pivot_thresh=1.0)
            t_fact = time.perf_counter() - t0
            t1 = time.perf_counter()
            dx = lu.solve(rhs)
            t_solve = time.perf_counter() - t1
            return dx, t_fact, t_solve, 'LU'
        elif self.solver == 'qr':
            if not HAS_SPARSEQR:
                raise RuntimeError("QR path requires 'sparseqr' (SuiteSparseQR). pip install sparseqr")
            # SuiteSparseQR factor/solve (handles rectangular too, but J is square)
            t0 = time.perf_counter()
            Q, R, E, rank = sparseqr.qr(J)  # factorization
            t_fact = time.perf_counter() - t0
            t1 = time.perf_counter()
            dx = sparseqr.solve(Q, R, E, rhs)
            t_solve = time.perf_counter() - t1
            return dx, t_fact, t_solve, 'QR'
        else:
            raise ValueError("solver must be 'lu' or 'qr'")

    def _apply_update(self, x, dx):
        # helper: update angles then PQ magnitudes from a concatenated dx
        nang = len(self.ang_idx)
        th_trial = x["th"].copy()
        V_trial  = x["V"].copy()
        th_trial[self.ang_idx] += dx[:nang]
        V_trial[self.pq]       += dx[nang:]
        # keep voltage magnitudes physical
        V_trial[self.pq] = np.clip(V_trial[self.pq], 0.5, 1.6)
        return th_trial, V_trial

    # ----- main NR loop -----
    def run(self, verbose=True):
        hist = []
        for it in range(1, self.max_it+1):
            mis, P, Q = self.mismatch()
            norm0 = float(np.max(np.abs(mis)))
            if verbose:
                print(f"[NR] it={it:02d}  ||m||_inf={norm0:.3e}")
            if norm0 < self.tol:
                break

            tJ0 = time.perf_counter()
            J = self.jacobian(P, Q)
            tJ = time.perf_counter() - tJ0

            dx, t_fact, t_solve, method = self.solve_linear(J, mis)

            # --- backtracking line search ---
            x0 = {"th": self.th, "V": self.V}
            alpha = 1.0
            improved = False
            for _ in range(8):  # up to 8 backtracks
                th_try, V_try = self._apply_update(x0, alpha*dx)
                th_save, V_save = self.th, self.V
                self.th, self.V = th_try, V_try
                mis_try, _, _ = self.mismatch()
                norm_try = float(np.max(np.abs(mis_try)))
                if norm_try < norm0:  # accept
                    improved = True
                    break
                # revert and shrink step
                self.th, self.V = th_save, V_save
                alpha *= 0.5
            if not improved:
                # last resort: take tiny step
                th_try, V_try = self._apply_update(x0, 0.1*dx)
                self.th, self.V = th_try, V_try

            hist.append({
                "iter": it, "mismatch_inf": norm0,
                "t_jacobian": tJ, "t_fact": t_fact, "t_solve": t_solve, "method": method
            })
        return {
            "converged": (norm0 < self.tol),
            "iterations": it,
            "V": self.V, "theta": self.th,
            "history": hist
        }
