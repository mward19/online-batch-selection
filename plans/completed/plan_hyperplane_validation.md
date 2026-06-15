# Plan: Hyperplane Validation Notebook

Create `scripts/blobs-hyperplane-validation.ipynb` to visually confirm that the ground-truth hyperplane (`w*`) and the noised teacher hyperplane (`w_hat`) used by RhoLoss are sensibly related to the dataset.

---

## Available artifacts

All in `models/teacher/`:
| File | Shape | Contents |
|---|---|---|
| `makeblobs_1024d_centers_seed42.npy` | (2, 1024) | Cluster centers fed to `make_blobs` |
| `makeblobs_1024d_wstar_seed42.npy` | (1024,) | Ground-truth unit direction `w*` |
| `makeblobs_1024d_wnoised_alpha0.5_nseed0.npy` | (1024,) | Noised unit direction `w_hat` (alpha=0.5, θ≈27°) |

---

## Approach: project onto 2D subspace spanned by (w\*, w\_hat)

The data is 1024-dimensional but the interesting structure lives in the plane defined by `w*` and `w_hat`. Use Gram-Schmidt to build an orthonormal basis for that plane:

```
e1 = w*
e2 = normalize(w_hat - (w_hat · w*) * w*)
```

Project every data point `x` → `(x·e1, x·e2)`. In this coordinate system:
- The `w*` decision boundary is the line perpendicular to `(1, 0)` through the origin → vertical line `x1 = 0`
- The `w_hat` decision boundary is the line perpendicular to `(cos θ, sin θ)` through the origin → line with slope `-cos θ / sin θ`

where `θ = arccos(w* · w_hat)`.

---

## Notebook cells

### Cell 1 — Imports
```python
import sys
sys.path.append('..')

import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
```

### Cell 2 — Load teacher artifacts
```python
TEACHER_DIR = '../models/teacher'

centers = np.load(f'{TEACHER_DIR}/makeblobs_1024d_centers_seed42.npy')   # (2, 1024)
w_star  = np.load(f'{TEACHER_DIR}/makeblobs_1024d_wstar_seed42.npy')      # (1024,)
w_hat   = np.load(f'{TEACHER_DIR}/makeblobs_1024d_wnoised_alpha0.5_nseed0.npy')  # (1024,)

cos_theta = float(np.dot(w_star, w_hat))
theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
print(f'cos θ = {cos_theta:.4f},  θ = {theta_deg:.1f}°')
```

### Cell 3 — Regenerate dataset
```python
# Match params in configs/makeblobs/data/makeblobs_1024d_2class.yaml
# seed matches config seed=1 (random_state falls back to config seed)
X, y = make_blobs(
    n_samples=10_000,
    n_features=1024,
    centers=centers,
    cluster_std=1.0,
    random_state=1,
)
X = X.astype(np.float32)
print(f'X shape: {X.shape}, classes: {np.unique(y)}')
```

### Cell 4 — Build 2D projection basis
```python
e1 = w_star / np.linalg.norm(w_star)
e2_raw = w_hat - np.dot(w_hat, e1) * e1
e2 = e2_raw / np.linalg.norm(e2_raw)

X_proj = X @ np.stack([e1, e2], axis=1)   # (N, 2)
```

### Cell 5 — Plot
```python
fig, ax = plt.subplots(figsize=(7, 6))

for cls, label, color in [(0, 'Class 0', 'steelblue'), (1, 'Class 1', 'tomato')]:
    mask = y == cls
    ax.scatter(X_proj[mask, 0], X_proj[mask, 1],
               alpha=0.15, s=5, color=color, label=label)

# Decision boundaries: lines through origin perpendicular to each direction
xlim = ax.get_xlim() or (-4, 4)
t = np.linspace(-5, 5, 300)

# w* boundary: perpendicular to e1=(1,0) → vertical line x1=0
ax.axvline(0, color='green', lw=2, label='w* boundary (optimal)')

# w_hat boundary: perpendicular to (cos θ, sin θ) → slope = -cos θ / sin θ
sin_theta = np.sqrt(1 - cos_theta**2)
slope = -cos_theta / sin_theta if sin_theta > 1e-8 else np.inf
if np.isfinite(slope):
    ax.plot(t, slope * t, color='orange', lw=2, ls='--',
            label=f'w_hat boundary (θ={theta_deg:.1f}°)')

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel('projection onto w*')
ax.set_ylabel('projection onto w_hat⊥')
ax.set_title('MakeBlobs 1024-D: projected onto (w*, w_hat) plane')
ax.legend()
plt.tight_layout()
plt.show()
```

{{Add a Cell 6 after the plot that computes classification accuracy for both w* and w_hat on the full dataset. The sign of `x·w` determines the predicted class (positive → class 1, negative → class 0). Report both accuracies side by side so you can see how much the noise costs.}}

### Cell 6 — Classifier accuracy
```python
# Sign of projection onto w gives predicted class (class 1 if positive, class 0 if negative)
pred_wstar = (X @ w_star >= 0).astype(int)
pred_what  = (X @ w_hat  >= 0).astype(int)

acc_wstar = (pred_wstar == y).mean()
acc_what  = (pred_what  == y).mean()

print(f'w*   accuracy: {acc_wstar:.4f}')
print(f'w_hat accuracy: {acc_what:.4f}')
print(f'accuracy gap:  {acc_wstar - acc_what:.4f}')
```

---

## Files

- [x] ~~`notebooks/blobs-hyperplane-validation.ipynb`~~
