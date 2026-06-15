# Plan: Log ground-truth hyperplane accuracy to W&B

## Background

Each sample $\mathbf{x} \in \mathbb{R}^d$ is drawn from a class-conditional Gaussian:

$$\mathbf{x} \mid \text{class } k \sim \mathcal{N}(\boldsymbol{\mu}_k,\ I_d), \quad \boldsymbol{\mu}_0 = +d\cdot \mathbf{w}^*,\quad \boldsymbol{\mu}_1 = -d\cdot \mathbf{w}^*$$

where $\|\mathbf{w}^*\| = 1$, `cluster_std` = 1, and $d$ is the center scale (`cscale`) parameter determining how far from the origin the cluster centers are. The Bayes-optimal rule for equal priors and shared spherical covariance reduces to a threshold on the scalar projection $z = \mathbf{x} \cdot \mathbf{w}^* \in \mathbb{R}$: decide class 0 iff $z > 0$.

**Distribution of $z$.**  For any linear projection $z = \mathbf{w}^{*\top} \mathbf{x}$ with $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, I_d)$:

$$z \sim \mathcal{N}(\mathbf{w}^{*\top}\boldsymbol{\mu},\ \mathbf{w}^{*\top} I_d\, \mathbf{w}^*) = \mathcal{N}(\mathbf{w}^{*\top}\boldsymbol{\mu},\ \|\mathbf{w}^*\|^2) = \mathcal{N}(\mathbf{w}^{*\top}\boldsymbol{\mu},\ 1)$$

{{This is the standard result that any affine function of a Gaussian is Gaussian (see e.g. [Bishop PRML §2.3](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) or the [Wikipedia multivariate normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Affine_transformation)). A quick proof: write $\mathbf{x} = \boldsymbol{\mu} + \boldsymbol{\varepsilon}$ where $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I_d)$, so $z = \mathbf{w}^{*\top}\boldsymbol{\mu} + \mathbf{w}^{*\top}\boldsymbol{\varepsilon}$. The second term is $\sum_i w^*_i \varepsilon_i$ — a sum of independent normals — which is $\mathcal{N}(0,\ \|\mathbf{w}^*\|^2)$ by the standard sum-of-normals rule. Hence $z$ is Gaussian with the stated mean and variance.}}

So:

- class 0: $z \sim \mathcal{N}(+d,\ 1)$
- class 1: $z \sim \mathcal{N}(-d,\ 1)$

**Error integrals.**

$$P(\text{error} \mid \text{class 0}) = P(z < 0 \mid \text{class 0}) = \int_{-\infty}^{0} \mathcal{N}(z;\ {+}d,\ 1)\, dz = \Phi\!\left(\frac{0 - d}{1}\right) = \Phi(-d)$$

$$P(\text{error} \mid \text{class 1}) = P(z > 0 \mid \text{class 1}) = \int_{0}^{\infty} \mathcal{N}(z;\ {-}d,\ 1)\, dz = 1 - \Phi\!\left(\frac{0-(-d)}{1}\right) = 1 - \Phi(d) = \Phi(-d)$$

where $\Phi(x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} e^{-t^2/2}\, dt$ is the standard normal CDF, and the last step uses $1 - \Phi(a) = \Phi(-a)$. Both conditional errors are equal, so with equal class priors:

$$\text{bayes\_error} = \Phi(-d), \qquad \text{bayes\_accuracy} = 1 - \Phi(-d) = \Phi(d)$$

The geometry: class 0's mean is at $z = +d$, the decision boundary is at $z = 0$ — a distance of $d/\sigma = d$ standard deviations away. The error is the tail mass on the wrong side. Larger `cscale` $\Rightarrow$ better-separated clusters $\Rightarrow$ smaller error.

Spot-check: $d = 1.0 \Rightarrow \Phi(1.0) \approx 0.841$, matching the notebook's empirical result.

[[Matthew's note: I have reviewed this math, and I think it checks out]]

---

## Option A — Inject into the generated data config (no code changes)

Add `bayes_accuracy` to the dict written by `write_generated_configs` in the slurm script:

```python
from scipy.special import ndtr  # Φ

cfg['bayes_accuracy'] = float(ndtr(cs))
```

If `main.py` logs the full merged config to W&B (likely — it saves `config.yaml` and most
frameworks log config as run metadata), this value appears automatically as a W&B config key.
No new code paths needed.

**Pros**: trivial to add, zero code changes, survives as part of the saved `config.yaml` per run.  
**Cons**: config keys in W&B are not time-series metrics — they appear in the run's "Config" panel,
not as a plotted line or summary stat. Fine for filtering/grouping; less useful if you want it
overlaid on a learning curve.

---

## Option B — Save metadata alongside the teacher, read at submission time

Extend `make_blobs_teacher.py` to write a sidecar JSON:
```
models/teacher/makeblobs_{d}d_cscale{cs}_meta.json
  { "bayes_accuracy": 0.841, "bayes_error": 0.159, "theta_deg": 44.8, "cscale": 1.0, "dim": 1024 }
```

Then `write_generated_configs` reads it and injects into the data config (same as Option A from
that point on).

**Pros**: all geometry stats in one place; reusable by notebook, analysis scripts, etc.  
**Cons**: requires a small code change to `make_blobs_teacher.py` (plan + implement separately).
Intermediate file to manage.

---

## Option C — Log as a W&B summary metric at run start via diagnostics

Add a one-shot `wandb.summary["bayes_accuracy"]` call inside the diagnostics logger at epoch 0,
reading `bayes_accuracy` out of the config (put there by Option A or B). Summary metrics appear
prominently in the W&B runs table and can be used for coloring/sorting, unlike config keys.

**Pros**: shows up in the W&B runs table as a sortable column alongside val_acc.  
**Cons**: requires a code change to `DiagnosticsLogger` (needs a plan). Still depends on
Option A or B to get the value into the config first.

---

## Option D — Compute empirically on the test set using w_star

Load `w_star` from `models/teacher/makeblobs_{d}d_cscale{cs}_wstar_seed42.npy` inside the
training code and evaluate its accuracy on the test set each epoch (or once at init). Log as
`wstar_test_acc` to W&B.

**Pros**: empirical — captures finite-sample and standardization effects; gives a meaningful
reference line directly on the val-accuracy plot.  
**Cons**: biggest code change of the options (touches the data loader or `SelectionMethod`);
needs `w_star` accessible at training time.

---

## Recommendation

**Option A** first — it's a one-liner in the slurm script, zero risk, and gets the value into
W&B config immediately. If you later want it as a runs-table column or overlaid on a plot,
add **Option C** on top (small diagnostics change). Option D is the richest but most invasive;
worth it only if you want per-epoch reference lines.
