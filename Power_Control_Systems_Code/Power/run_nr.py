import numpy as np
import scipy.sparse as sp
from nr_powerflow_solvers import NRSolver

# --- Consistent 3-bus Ybus (reactive lines) ---
# Lines: x12=0.20, x13=0.25, x23=0.40 (r=0)  →  y = 1/(j x) = -j/x
y12 = -1j * (1/0.20)   # -j*5
y13 = -1j * (1/0.25)   # -j*4
y23 = -1j * (1/0.40)   # -j*2.5

# Correct Ybus construction:
Y = sp.csr_matrix([
    [ (y12 + y13),   -y12,          -y13        ],
    [    -y12,     (y12 + y23),     -y23        ],
    [    -y13,        -y23,       (y13 + y23)   ],
], dtype=complex)

# (Numerically:
#  [[ -0-9.0j,  0+5.0j,  0+4.0j],
#   [  0+5.0j, -0-7.5j,  0+2.5j],
#   [  0+4.0j,  0+2.5j, -0-6.5j ]] )


# Bus types: 0=Slack, 1=PV, 2=PQ
bus_type = np.array([0, 2, 2])   # bus 1 slack, buses 2 & 3 PQ

# Net injections (gen - load), per-unit. (Loads are positive consumption.)
# Here we specify only loads at PQ buses; slack will supply the rest.
P_spec = np.array([0.0, -0.50, -0.30])   # P2=0.50 pu load, P3=0.30 pu load
Q_spec = np.array([0.0, -0.20, -0.10])   # Q2=0.20 pu load, Q3=0.10 pu load

# Flat start
V0  = np.array([1.0, 1.0, 1.0])
th0 = np.array([0.0, 0.0, 0.0])

# Run with LU and a line search (see patch below)
solver = NRSolver(Y, bus_type, P_spec, Q_spec, V0, th0,
                  solver='lu', reordering='COLAMD')
res = solver.run(verbose=True)

print("\nConverged:", res["converged"])
print("Iterations:", res["iterations"])
print("Final V:", np.round(res["V"], 6))
print("Final θ (deg):", np.round(np.degrees(res["theta"]), 4))
