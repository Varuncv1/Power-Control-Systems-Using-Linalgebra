# Power-Control-Systems-Using-Linalgebra

# SVD-based Abrupt Change Detection

This repository implements the abrupt change detection heuristic based on the **Singular Value Decomposition (SVD)** of a *history matrix of differences*, following Sanandaji et al.

---

## What this code does

* Builds, at each time step *t*, a **history matrix of differences**

  $\Delta_t = \big[\, y_t{-}y_{t-1},\ y_t{-}y_{t-2},\ \ldots,\ y_t{-}y_{t-w} \,\big] \in \mathbb{R}^{M\times w},$

  where $y_t$ stacks the $M$ measurement channels at time *t* and $w$ is the window length.
* Computes the **largest singular value** $\sigma_1(\Delta_t)$.
* Uses a **robust baseline threshold** (from a pre-change segment) to flag abrupt changes when $\sigma_1(\Delta_t)$ spikes above the threshold.
* Provides a synthetic demo and CSV-driven workflow, optional per-channel z-scoring, plotting, and CSV export of detection results.

Typical outputs:

* A time series of $\sigma_1(\Delta_t)$ and the threshold.
* A boolean detection mask indicating change points.
* An illustrative plot saved as `*_sigma1_paper.png` when `--save-prefix` is provided.

---

## The linear algebra method we solve

* **Method:** Singular Value Decomposition (SVD) of the difference history matrix $\Delta_t$.
* **Detection statistic:** the **spectral norm** (largest singular value) $\sigma_1(\Delta_t)$.
* **Rationale:** An abrupt change behaves like an additive **low-rank (often rank-1) perturbation** to $\Delta_t$, creating a sharp increase in $\sigma_1$. Monitoring this scalar over time detects change points robustly even in the presence of bounded state variation and Gaussian noise (as analyzed in the referenced paper).

---

## Paper this implementation follows

Sanandaji, B. M., Bitar, E., Poolla, K., & Vincent, T. L. (2014). *An Abrupt Change Detection Heuristic with Applications to Cyber Data Attacks on Power Systems*. arXiv:1404.1978.

> The code mirrors the paper’s construction of $\Delta_t$, tracks $\sigma_1(\Delta_t)$, and uses a robust baseline to decide when a jump indicates a change.


## Quick start

### Synthetic demo

```bash
python svd.py --demo --win 16 --baseline 200 --k 6 --method mad
```

## Citation

If you use this code, please cite the paper and acknowledge this implementation:

```
@article{Sanandaji2014ACD,
  title={An Abrupt Change Detection Heuristic with Applications to Cyber Data Attacks on Power Systems},
  author={Sanandaji, Borhan M. and Bitar, Eilyan and Poolla, Kameshwar and Vincent, Tyrone L.},
  journal={arXiv:1404.1978},
  year={2014}
}
```

# Load Frequency Control with Hierarchical Bi-Level Approach

## What the code does

The provided Python code (`load_freq.py`) simulates **Load Frequency Control (LFC)** in a multi-area interconnected power system. It models system frequency deviations, tie-line power flows, and applies controllers to maintain system stability under disturbances. Specifically, it implements a **hierarchical control strategy**, where proportional–integral (PI) controllers operate at the lower level and are tuned/optimized, while a supervisory control layer guides these PI controllers using optimal control principles.

---

## The linear algebra method we are solving

The code applies the **Linear Quadratic Regulator (LQR)** method for optimal control design. LQR solves a **quadratic optimization problem** on the state-space representation of the interconnected power system:

$$
\min J = \int (x^T Q x + u^T R u) dt
$$

subject to the linear state-space model $\dot{x} = Ax + Bu$.

The LQR provides optimal state-feedback gains $K$, which set reference signals for the lower-level PI controllers. A **Kalman filter** is also used for state estimation when not all states are measurable.

---

## The paper this implementation is based on

This implementation is based on the paper:

**Abdullahi Bala Kunya (2024).** *Hierarchical bi-level load frequency control for multi-area interconnected power systems*. International Journal of Electrical Power and Energy Systems, 155, 109600. \[DOI: 10.1016/j.ijepes.2023.109600]【28†files\_uploaded\_in\_conversation】

The paper proposes a **supervisory LQR–PI control scheme** for multi-area LFC, demonstrating improved performance in stability, robustness, and disturbance handling compared to classical PI-only approaches.
