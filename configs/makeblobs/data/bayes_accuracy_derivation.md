# Bayes accuracy derivation for MakeBlobs

`bayes_accuracy` is computed analytically and injected into each generated data config by
`write_generated_configs` in `slurm_run_blobs_deep_linear.py` as `float(ndtr(cscale))`.
Derivation below.

---

Each sample $\mathbf{x} \in \mathbb{R}^d$ is drawn from a class-conditional Gaussian:

$$\mathbf{x} \mid \text{class } k \sim \mathcal{N}(\boldsymbol{\mu}_k,\ I_d), \quad \boldsymbol{\mu}_0 = +d\cdot \mathbf{w}^*,\quad \boldsymbol{\mu}_1 = -d\cdot \mathbf{w}^*$$

where $\|\mathbf{w}^*\| = 1$, `cluster_std` = 1, and $d$ is the center scale (`cscale`) parameter determining how far from the origin the cluster centers are. The Bayes-optimal rule for equal priors and shared spherical covariance reduces to a threshold on the scalar projection $z = \mathbf{x} \cdot \mathbf{w}^* \in \mathbb{R}$: decide class 0 iff $z > 0$.

**Distribution of $z$.**  For any linear projection $z = \mathbf{w}^{*\top} \mathbf{x}$ with $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, I_d)$:

$$z \sim \mathcal{N}(\mathbf{w}^{*\top}\boldsymbol{\mu},\ \mathbf{w}^{*\top} I_d\, \mathbf{w}^*) = \mathcal{N}(\mathbf{w}^{*\top}\boldsymbol{\mu},\ \|\mathbf{w}^*\|^2) = \mathcal{N}(\mathbf{w}^{*\top}\boldsymbol{\mu},\ 1)$$

This is the standard result that any affine function of a Gaussian is Gaussian (see e.g. [Bishop PRML §2.3](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) or the [Wikipedia multivariate normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Affine_transformation)). A quick proof: write $\mathbf{x} = \boldsymbol{\mu} + \boldsymbol{\varepsilon}$ where $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I_d)$, so $z = \mathbf{w}^{*\top}\boldsymbol{\mu} + \mathbf{w}^{*\top}\boldsymbol{\varepsilon}$. The second term is $\sum_i w^*_i \varepsilon_i$ — a sum of independent normals — which is $\mathcal{N}(0,\ \|\mathbf{w}^*\|^2)$ by the standard sum-of-normals rule. Hence $z$ is Gaussian with the stated mean and variance.

So:

- class 0: $z \sim \mathcal{N}(+d,\ 1)$
- class 1: $z \sim \mathcal{N}(-d,\ 1)$

**Error integrals.**

$$P(\text{error} \mid \text{class 0}) = P(z < 0 \mid \text{class 0}) = \int_{-\infty}^{0} \mathcal{N}(z;\ {+}d,\ 1)\, dz = \Phi\!\left(\frac{0 - d}{1}\right) = \Phi(-d)$$

$$P(\text{error} \mid \text{class 1}) = P(z > 0 \mid \text{class 1}) = \int_{0}^{\infty} \mathcal{N}(z;\ {-}d,\ 1)\, dz = 1 - \Phi\!\left(\frac{0-(-d)}{1}\right) = 1 - \Phi(d) = \Phi(-d)$$

where $\Phi(x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} e^{-t^2/2}\, dt$ is the standard normal CDF, and the last step uses $1 - \Phi(a) = \Phi(-a)$. Both conditional errors are equal, so with equal class priors:

$$\text{bayes\_error} = \Phi(-d), \qquad \text{bayes\_accuracy} = 1 - \Phi(-d) = \Phi(d)$$

The geometry: class 0's mean is at $z = +d$, the decision boundary is at $z = 0$ — a distance of $d/\sigma = d$ standard deviations away. The error is the tail mass on the wrong side. Larger `cscale` $\Rightarrow$ better-separated clusters $\Rightarrow$ smaller error.

Spot-check: $d = 1.0 \Rightarrow \Phi(1.0) \approx 0.841$, matching empirical evaluation on the generated dataset.
