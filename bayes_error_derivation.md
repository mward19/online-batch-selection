# Bayes Error for Symmetric Gaussian Blobs

## Setup

Two equally-probable classes with isotropic Gaussian class-conditionals:

$$p(x \mid y=k) = \mathcal{N}(c_k,\, \sigma^2 I_d), \qquad k \in \{0, 1\}$$

Centers are symmetric: $c_0 = -c_1 = s \cdot w^*$, where $w^*$ is a unit vector and $s > 0$ is the center scale. Dimension is $d$.

---

## Bayes Optimal Classifier

The Bayes classifier assigns class 0 if $p(x \mid y=0) > p(x \mid y=1)$, i.e., if the log-likelihood ratio is positive:

$$\mathrm{LLR}(x) = \log \frac{p(x \mid y=0)}{p(x \mid y=1)} > 0$$

### Computing the LLR

For Gaussians with equal covariance $\sigma^2 I$:

$$\mathrm{LLR}(x) = -\frac{1}{2\sigma^2}\|x - c_0\|^2 + \frac{1}{2\sigma^2}\|x - c_1\|^2$$

Expanding the squared norms:

$$= \frac{1}{2\sigma^2}\left[\|x\|^2 - 2c_0 \cdot x + \|c_0\|^2 - \|x\|^2 + 2c_1 \cdot x - \|c_1\|^2\right]$$

The $\|x\|^2$ terms cancel. Since $\|c_0\| = \|c_1\| = s$ (both $\|c_k\|^2 = s^2$), those terms also cancel, leaving:

$$\mathrm{LLR}(x) = \frac{1}{\sigma^2}(c_0 - c_1)\cdot x = \frac{2s}{\sigma^2}\, w^* \cdot x$$

**Key result**: the Bayes optimal decision depends only on $w^* \cdot x$, the projection onto the separating direction. All $d - 1$ orthogonal dimensions are irrelevant — they contribute identically to both likelihoods and cancel. The 1024-dimensionality does not affect Bayes error.

### Optimal boundary

Classify as $y = 0$ iff $w^* \cdot x > 0$.

---

## Bayes Error Rate

The 1D projection $z = w^* \cdot x$ is Gaussian:

$$z \mid y=0 \;\sim\; \mathcal{N}(+s,\, \sigma^2), \qquad z \mid y=1 \;\sim\; \mathcal{N}(-s,\, \sigma^2)$$

By symmetry of the two classes:

$$P(\text{error}) = P(z < 0 \mid y=0) = P\!\left(\mathcal{N}(+s, \sigma^2) < 0\right) = \Phi\!\left(\frac{0 - s}{\sigma}\right) = \Phi\!\left(-\frac{s}{\sigma}\right)$$

### General formula

$$\boxed{P(\text{Bayes error}) = \Phi\!\left(-\frac{\texttt{center\_scale}}{\texttt{cluster\_std}}\right)}$$

### Values for this experiment

With `center_scale = 1.0` and `cluster_std = 1.0`:

$$P(\text{error}) = \Phi(-1) \approx 15.87\%$$

| cluster\_std | separation $s/\sigma$ | $P(\text{error})$ |
|---|---|---|
| 0.5 | 2.0 | 2.3% |
| 1.0 | 1.0 | 15.9% |
| 1.5 | 0.67 | 25.2% |
| 2.0 | 0.5 | 30.9% |

---

## Teacher Noise Analysis

The noised teacher uses $\hat{w} = (w^* + \alpha / \sqrt{d} \cdot \varepsilon) / \|\cdot\|$ where $\varepsilon \sim \mathcal{N}(0, I_d)$ and $\alpha$ is a dimensionless noise parameter.

### Why $1/\sqrt{d}$ is the natural scale

$\varepsilon \sim \mathcal{N}(0, I_d)$ has expected norm $E[\|\varepsilon\|] = \sqrt{d}$, so the noise vector $\varepsilon / \sqrt{d}$ has expected norm 1 — matching $\|w^*\| = 1$. The parameter $\alpha$ therefore directly controls the noise-to-signal ratio in terms of vector magnitudes.

### Angle between $\hat{w}$ and $w^*$

Before normalization: $\tilde{w} = w^* + \frac{\alpha}{\sqrt{d}} \varepsilon$. The squared norm concentrates around:

$$\|\tilde{w}\|^2 \approx \|w^*\|^2 + \frac{\alpha^2}{d}\|\varepsilon\|^2 \approx 1 + \alpha^2$$

The dot product $\tilde{w} \cdot w^* = 1 + \frac{\alpha}{\sqrt{d}}(\varepsilon \cdot w^*) \approx 1$ (cross term mean 0, negligible variance). So:

$$\cos\theta \approx \frac{\tilde{w} \cdot w^*}{\|\tilde{w}\|} \approx \frac{1}{\sqrt{1 + \alpha^2}}$$

| $\alpha$ | $\cos\theta$ | $\theta$ | noise\_std (d=1024) |
|---|---|---|---|
| 0.1 | 0.995 | 5.7° | 0.003 |
| 0.5 | 0.894 | 26.6° | 0.016 |
| 1.0 | 0.707 | 45.0° | 0.031 |
| 2.0 | 0.447 | 63.4° | 0.063 |
| 10 | 0.100 | 84.3° | 0.313 |

Note: $\texttt{noise\_std} = \alpha / \sqrt{d}$.
