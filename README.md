# Power-Control-Systems-Using-Linalgebra

# 1) SVD-based Abrupt Change Detection

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

# 2) Load Frequency Control with Hierarchical Bi-Level Approach

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


# 3) Newton–Raphson Power Flow Solver

## What the code does

The provided Python scripts (`nr_powerflow_solvers.py` and `run_nr.py`) implement a **Newton–Raphson (NR) based power flow solver** for electric power systems. The solver computes the steady-state operating point of a transmission network by iteratively solving the nonlinear algebraic equations governing active and reactive power balances at each bus. The code:

* Builds bus admittance matrices.
* Defines mismatch equations for power balance.
* Iteratively applies the NR method to update bus voltage magnitudes and phase angles until convergence.
* Provides results for bus voltages, power injections, and line flows.

---

## The linear algebra method we are solving

The underlying method is the **Newton–Raphson method applied to nonlinear power flow equations**. At each iteration, the following linear system is solved:

$$
J(x_k) \Delta x = -F(x_k)
$$

where:

* $F(x_k)$ is the mismatch vector of active/reactive power equations,
* $J(x_k)$ is the Jacobian matrix of partial derivatives,
* $\Delta x$ is the correction to the state vector (bus voltage magnitudes and angles).

This reduces the nonlinear power flow problem into repeated solutions of linear systems until the mismatches are within tolerance.

---

## The paper this implementation is based on

This solver and its design follow the exposition in:

**Monticelli, A. (1985).** *State Estimation in Electric Power Systems: A Generalized Approach*. KTH Royal Institute of Technology. \[Full text PDF provided]【37†files\_uploaded\_in\_conversation】

The Newton–Raphson formulation in the code aligns with the standard derivation and methodology discussed in Monticelli’s work on power system state estimation and load flow analysis.


# 4) Stochastic Subspace Identification for Power System Stability

## What the code does

The provided Python code (`ssi_cva_power.py`) implements **stochastic subspace identification (SSI)** using the canonical variate algorithm (CVA) for power systems. It processes time-series measurement data (e.g., generator powers, bus voltages) under small random load variations to identify critical oscillatory modes. These identified modes are then used to compute a **stability index (SISI)** that predicts the proximity of the system to oscillatory instability without requiring large disturbances.

---

## The linear algebra method we are solving

The algorithm is based on **subspace identification techniques**, which make heavy use of:

* **Block Hankel matrices** to organize past and future measured outputs.
* **QR factorization and Singular Value Decomposition (SVD)** to extract system order and dominant modes.
* **Least-squares solutions** for estimating state-space matrices ($A_d, C$).

At its core, the method reduces to repeatedly applying SVD to projection matrices to determine the system’s modal content and then computing eigenvalues of the identified state matrix. These eigenvalues are used to evaluate damping ratios and predict Hopf bifurcation points.

---

## The paper this implementation is based on

This implementation is based on the paper:

**Hassan Ghasemi, Claudio Cañizares, and Ali Moshref (2006).** *Oscillatory Stability Limit Prediction Using Stochastic Subspace Identification.* IEEE Transactions on Power Systems. Accepted September 2005【47†revised.dvi】.

The paper introduces the use of stochastic subspace identification with CVA to extract critical modes directly from measured ambient signals and defines a **System Identification Stability Index (SISI)** for online stability monitoring.


# 5) Lyapunov Functions for Multimachine Power Systems

## What the code does

The provided Python code (`davy_hiskens_energy_demo.py`) demonstrates the use of **Lyapunov (energy) functions** for assessing the stability of multimachine power systems with dynamic loads. It models generator swing dynamics, load recovery behavior, and applies direct energy function methods to evaluate system stability following disturbances. The code illustrates how energy margins can be computed and compared against critical energy thresholds to predict whether the system will remain stable or lose synchronism.

---

## The linear algebra method we are solving

The approach is grounded in the construction and evaluation of **Lyapunov functions** derived via:

* **First integral analysis** for formulating candidate energy functions.
* **Popov stability criterion** leading to Lyapunov functions of the Lurè–Postnikov form.
* Use of **Jacobian matrices** of the power flow equations to verify positive definiteness conditions ensuring local stability.

These steps reduce stability analysis to evaluating quadratic forms and integral terms that represent stored system energy, ensuring that the Lyapunov function is positive definite and non-increasing along system trajectories.

---

## The paper this implementation is based on

This implementation is based on the paper:

**Robert J. Davy and Ian A. Hiskens (1997).** *Lyapunov Functions for Multimachine Power Systems with Dynamic Loads.* IEEE Transactions on Circuits and Systems I: Fundamental Theory and Applications, 44(9), 796–812【56†lyapunov-functions-for-multimachine-power-systems-with-4t2dhjgrs1.pdf】.

The paper develops strict Lyapunov functions for multimachine power systems with dynamic reactive power loads, extending traditional energy function methods to incorporate load dynamics and enabling direct stability assessment of both angle and voltage dynamics.



# 6) PMU-based Generator Parameter Estimation

## What the code does

The provided Python code (`pmu_wls_generator_params.py`) estimates synchronous generator parameters such as inertia constant, damping factor, turbine-governor time constant, and primary frequency control droop. It uses high-resolution phasor measurement unit (PMU) data from the generator terminal bus. The code constructs an autoregressive with exogenous input (ARX) model from input/output PMU data and applies linear estimation techniques to recover generator parameters. Case studies can be run on simulated or real PMU datasets.

---

## The linear algebra method we are solving

The estimation problem is formulated as a **linear least squares** problem. After discretizing the continuous generator transfer function into an ARX model (using Zero-Order Hold or Tustin’s method), the estimation reduces to solving:

$$
\mathbf{b} = \mathbf{A}\mathbf{x} + e
$$

where:

* $\mathbf{A}$ is the regression matrix built from lagged input/output data,
* $\mathbf{x}$ contains the ARX model coefficients,
* $\mathbf{b}$ is the vector of outputs,
* $e$ is the error vector.

The parameter vector $\mathbf{x}$ is solved via the normal equations:

$$
\mathbf{x} = (A^T A)^{-1} A^T b
$$

From the estimated coefficients, physical generator parameters ($H, R, T, D$) are recovered.

---

## The paper this implementation is based on

This implementation follows the methodology in:

**Bander Mogharbel, Lingling Fan, and Zhixin Miao (2015).** *Least Squares Estimation-Based Synchronous Generator Parameter Estimation Using PMU Data.* arXiv:1503.05224【65†source】.

The paper develops ARX-based linear least squares estimation for identifying generator parameters from PMU data, demonstrating both simulation and real-world case studies.



# 7) Power Flow Solvers

## What the Code Does
This repository provides implementations of numerical solvers for **AC power flow analysis**.  
The scripts compute steady-state operating conditions of electrical power systems, focusing on bus voltages, line flows, and power injections. The solvers are designed to test and compare conventional Newton-Raphson and alternative formulations for robustness and efficiency in solving nonlinear, nonconvex power flow problems.

## Linear Algebra Method
The primary linear algebra approach implemented here is the **Kronecker product–based bilinear least squares formulation**.  
This method reformulates the nonlinear AC power flow equations into a bilinear least-squares optimization problem, enabling efficient and stable iterative updates. It addresses limitations of the Newton-Raphson method by avoiding truncation errors and enhancing convergence properties.

## Reference Paper
The methodology and algorithms in this code are based on:

HyungSeon Oh,  
**"A Unified and Efficient Approach to Power Flow Analysis,"**  
*Energies*, vol. 12, no. 12, 2425, 2019.  
[https://doi.org/10.3390/en12122425](https://doi.org/10.3390/en12122425) :contentReference[oaicite:0]{index=0}



# 8) Phasor Measurement-Based State Estimation

## What the code does

The provided Python code (`phasor.py`) implements algorithms for **phasor measurement-based dynamic state estimation** in electric power systems. It processes phasor measurement unit (PMU) data to reconstruct system states, estimate generator dynamic variables, and improve monitoring of transient stability. The implementation demonstrates how synchronized PMU measurements can be used for enhanced situational awareness compared to traditional SCADA-based methods.

---

## The linear algebra method we are solving

The state estimation algorithm is formulated as a **weighted least squares (WLS)** optimization problem. At each iteration, the method solves:

$$
( H^T R^{-1} H ) \Delta x = H^T R^{-1} ( z - h(x) )
$$

where:

* $z$ is the vector of PMU measurements,
* $h(x)$ is the nonlinear measurement function,
* $H$ is the Jacobian of $h(x)$,
* $R$ is the covariance matrix of measurement errors,
* $\Delta x$ is the state correction.

This formulation reduces the estimation problem to repeated solutions of linear systems using Jacobian updates until convergence.

---

## The paper this implementation is based on

This implementation is based on the paper:

**Abur, A., & Exposito, A. G. (2010).** *Power System State Estimation: Theory and Implementation.* IEEE Press. (Referenced in IEEE paper 05871327 on PMU-based state estimation)【80†05871327.pdf】.

The paper describes the adaptation of weighted least squares state estimation for use with synchronized phasor measurements, providing the theoretical foundation for the code.



# 9) Graph Signal Processing Based State Estimation

## What the Code Does

This code implements power system state estimation under partial observability using **graph signal processing (GSP)** techniques. It provides algorithms for recovering system states (voltage magnitudes and phases) when the number of sensors is insufficient for conventional methods. The implementation also includes graph-based regularized weighted least squares (GSP-WLS) solvers and sensor placement strategies.

---

## Linear Algebra Method

The core linear algebra method being solved here is a **regularized weighted least squares (WLS)** optimization problem with **graph Laplacian-based smoothness constraints**.

The standard WLS formulation is:

$$
\min_x (z - Hx)^T R^{-1} (z - Hx)
$$

where:

* $z$ is the measurement vector,
* $H$ is the measurement matrix,
* $R$ is the covariance of measurement errors,
* $x$ is the state vector.

The GSP-WLS modifies this by adding a smoothness prior using the graph Laplacian $L$:

$$
\min_x (z - Hx)^T R^{-1} (z - Hx) + \mu x^T L x
$$

where $\mu > 0$ controls the weight of the graph regularization. The additional quadratic term enforces that estimated states vary smoothly across the network graph, improving estimation under limited observability.

This leads to solving linear systems of the form:

$$
(H^T R^{-1} H + \mu L) x = H^T R^{-1} z
$$

---

## Reference Paper

This implementation is based on the paper:

> Lital Dabush, Ariel Kroizer, Tirza Routtenberg,
> **“State Estimation in Partially Observable Power Systems via Graph Signal Processing Tools,”**
> *Sensors*, 23(3), 1387, 2023.
> [https://doi.org/10.3390/s23031387](https://doi.org/10.3390/s23031387)【90†sensors-23-01387-v2.pdf†L1-L20】




# 10) Small-Signal Stability Analysis with Primal and Dual GEPs

## What the Code Does
This code implements algorithms for small-signal stability analysis of power systems using **generalized eigenvalue problems (GEPs)**. It provides functionality to analyze system dynamics and compute eigenvalues/eigenvectors that determine oscillatory stability margins.

## Linear Algebra Method
The core linear algebra method being solved here is the **Generalized Eigenvalue Problem (GEP)** in both its **primal** and **dual** forms:

- **Primal GEP:**  
  \( Av = sBv \)

- **Dual GEP:**  
  \( B\hat{v} = \hat{s}A\hat{v} \)

where \(A\) and \(B\) are large, sparse, non-Hermitian matrices arising from linearized power system differential-algebraic equations (DAEs).

## Reference Paper
This implementation is based on the work:

**F. Milano and I. Dassios,**  
*“Primal and Dual Generalized Eigenvalue Problems for Power Systems Small-Signal Stability Analysis,”*  
IEEE Transactions on Power Systems, vol. 32, no. 6, pp. 4529–4540, Nov. 2017.  
DOI: [10.1109/TPWRS.2017.2679128](https://doi.org/10.1109/TPWRS.2017.2679128) :contentReference[oaicite:0]{index=0}  





# 11) Control and Observability in Power Networks

## What the code does

The provided Python code (`Control_Observe.py`) demonstrates controllability and observability analysis of dynamical systems, with application to power networks. It constructs system state-space models and uses Kalman rank conditions to test whether the system is fully controllable and observable. The code also illustrates methods for selecting control inputs and measurement outputs to ensure desired system properties.

---

## The linear algebra method we are solving

The analysis relies on **rank tests of controllability and observability matrices**:

* **Controllability matrix:** $C = [B, AB, A^2B, \ldots, A^{n-1}B]$
* **Observability matrix:** $O = [C^T, (CA)^T, (CA^2)^T, \ldots, (CA^{n-1})^T]^T$

A system is controllable/observable if these matrices are full rank. The code computes these matrices and evaluates their ranks to determine controllability and observability.

---

## The paper this implementation is based on

This implementation is based on the paper:

**F. Pasqualetti, S. Zampieri, and F. Bullo (2012).** *Controllability Metrics, Limitations and Algorithms for Complex Networks.* arXiv:1203.0129【108†1203.0129v1.pdf】.

The paper develops theory and algorithms for controllability and observability in complex dynamical networks, providing the basis for the computations implemented here.

