# Plan: Fix Teacher Sign Bug

The teacher model's weight matrix has inverted class assignments relative to the dataset.

## Root cause

In `data/make_blobs_teacher.py`, class centers are stored as:
```python
centers = np.stack([+center_scale * w_star, -center_scale * w_star])
```
So sklearn assigns **label 0 to samples near +w_star** and **label 1 to samples near -w_star**.

But the teacher state dict is saved as:
```python
state_dict = {
    'fc.weight': torch.stack([-w_hat_t, +w_hat_t]),
    ...
}
```
This gives:
- logit[0] = -w_hat·x  (negative for x near +w_star)
- logit[1] = +w_hat·x  (positive for x near +w_star)

So for a class 0 sample (near +w_star), the teacher predicts class 1. **The teacher's class assignments are inverted**, causing RhoLoss to score easy samples as hard and vice versa.

## Fix

### 1. `data/make_blobs_teacher.py`

Swap the rows of `fc.weight` so class 0 scores high when x is near +w_star:

```python
state_dict = {
    'fc.weight': torch.stack([+w_hat_t, -w_hat_t]),   # was: [-w_hat_t, +w_hat_t]
    'fc.bias':   torch.zeros(2),
}
```

After this fix: logit[0] = +w_hat·x > 0 for class 0 samples → correct.

### 2. `slurm_run_blobs_deep_linear.py` — always regenerate teacher

The current sentinel only runs `make_blobs_teacher.py` if the centers `.npy` is missing. Since that file already exists, it would skip regeneration even after the sign fix. Remove the `if not Path(CENTERS_NPY).exists():` guard so the teacher generation always runs:

```python
# Before:
CENTERS_NPY = "models/teacher/makeblobs_1024d_centers_seed42.npy"
if not Path(CENTERS_NPY).exists():
    print("Teacher model not found — generating geometry and teacher...")
    subprocess.run([...], check=True)

# After:
print("Generating geometry and teacher...")
subprocess.run(
    [
        "python", "data/make_blobs_teacher.py",
        "--n_features", "1024",
        "--center_scale", "1.0",
        "--center_seed", "42",
        "--alpha", "0.5",
        "--noise_seed", "0",
        "--out_dir", "models/teacher",
    ],
    check=True,
)
```

The script is fast and deterministic (same seed → same output), so unconditional execution is safe. The `CENTERS_NPY` variable and its `Path` import are no longer needed.

### 3. `notebooks/blobs-hyperplane-validation.ipynb` — Cell 6

Fix the sign in the accuracy cell to match the correct convention:
```python
pred_wstar = (X @ w_star < 0).astype(int)   # was: >= 0
pred_what  = (X @ w_hat  < 0).astype(int)   # was: >= 0
```

After the fix, expected accuracy for w* ≈ 1 - Bayes error ≈ 84%.

---

## Files

- [x] ~~`data/make_blobs_teacher.py` — swap sign of fc.weight rows~~
- [x] ~~`slurm_run_blobs_deep_linear.py` — always run teacher generation (remove existence check)~~
- [x] ~~`notebooks/blobs-hyperplane-validation.ipynb` — fix Cell 6 sign~~
