# Plan: Log `what_test_acc` to W&B summary

## Goal

Log the test accuracy of the perturbed teacher direction `w_hat` to `wandb.summary` as `what_test_acc`. Mirrors `wstar_test_acc` exactly. Uses `(z <= 0)` as the prediction rule for the same reason (label 0 is near +w_hat direction).

---

## Changes

### 1. `slurm_run_blobs_deep_linear.py` — inject `wnoised_file` into generated data config

In `write_generated_configs`, after the `wstar_file` line, add:

```python
cfg['dataset']['wnoised_file'] = f"models/teacher/makeblobs_{d}d_cscale{cs}_wnoised_alpha1.0_nseed0.npy"
```

### 2. `data/makeblobs.py` — compute `what_test_acc` before standardization

After the `wstar_test_acc` block and before `if standardize:`, add:

```python
what_test_acc = None
wnoised_file = dcfg.get('wnoised_file')
if wnoised_file is not None:
    w_hat = np.load(wnoised_file).astype(np.float32)
    z = X_test @ w_hat
    preds = (z <= 0).astype(np.int64)
    what_test_acc = float((preds == y_test).mean())
```

Pass `what_test_acc` through `_makeblobs_output`:
- Add `what_test_acc=None` parameter to `_makeblobs_output`
- In its body: `if what_test_acc is not None: payload['what_test_acc'] = what_test_acc`
- Pass `what_test_acc=what_test_acc` at the call site in `MakeBlobs`

### 3. `methods/method_utils/diagnostics_context.py`

Add field:
```python
what_test_acc: float | None = None
```

### 4. `methods/SelectionMethod.py`

Pass the value when constructing the context:
```python
what_test_acc=self.data_info.get('what_test_acc'),
```

### 5. `methods/method_utils/diagnostics.py`

In `__init__`, store alongside `wstar_test_acc`:
```python
self.what_test_acc = context.what_test_acc
```

In the summary block:
```python
if self.what_test_acc is not None:
    wandb.summary['what_test_acc'] = self.what_test_acc
```
