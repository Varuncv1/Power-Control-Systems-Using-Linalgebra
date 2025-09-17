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

---

## File layout

* `svd.py` — main script (CLI) that builds $\Delta_t$, computes $\sigma_1$, thresholds, and plots.

---

## Quick start

### Synthetic demo

```bash
python svd.py --demo --win 16 --baseline 200 --k 6 --method mad
```

### CSV data

```bash
python svd.py \
  --csv your.csv \
  --time-col time \
  --channels V1 V2 I1 I2 \
  --win 16 \
  --baseline 200 \
  --k 6 \
  --method mad \
  --norm z \
  --save-prefix out
```

**Arguments (common):**

* `--win` / `-w` *(int)*: history window length $w$.
* `--baseline` *(int)*: number of initial pre-change samples used to learn a robust threshold.
* `--k` *(float)*: scale for robust thresholding (e.g., MAD×k or IQR×k).
* `--method` *(str)*: robust method (`mad` or `iqr`).
* `--norm` *(str)*: optional per-channel normalization; `z` = z-score using baseline.
* `--save-prefix` *(str)*: save plot(s) and outputs with this prefix.

**CSV-specific:**

* `--csv` *(path)*: input file (rows = time, columns = channels).
* `--time-col` *(str)*: name of timestamp column.
* `--channels` *(list)*: names of columns to use as measurement channels $y_t$.

---

## How the threshold works (paper-faithful)

1. Use the first `--baseline` samples (assumed pre-change) to compute $\sigma_1(\Delta_t)$.
2. Compute a robust location/scale (e.g., median and MAD or IQR) of these baseline values.
3. Form a threshold: $\text{thr} = \text{location} + k\,\text{scale}$.
4. For subsequent times, flag a detection when $\sigma_1(\Delta_t) > \text{thr}$.

This approximates the paper’s theoretical tail bounds with a practical, data-driven rule.

---

## Notes & assumptions

* Channels in `--channels` should be synchronized and sampled uniformly.
* The paper’s analysis assumes **bounded state variation** and **Gaussian measurement noise**; the heuristic remains useful more generally but guarantees rely on those assumptions.
* Larger `--win` often improves sensitivity under higher noise, but may increase delay.

---

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

---

## License

MIT (unless otherwise specified).
