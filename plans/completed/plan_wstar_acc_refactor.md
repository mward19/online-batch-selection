# Plan: Refactor `wstar_test_acc` computation in `data/makeblobs.py`

## Problem

`w_star` is defined in raw (un-standardized) feature space — the cluster centers are placed at `±w_star` before any standardization occurs. The `wstar_test_acc` block (lines 149–161) runs after `X_test` has already been overwritten with its standardized form (line 142), so the code tries to compensate with a coordinate transformation. That transformation has a sign error on the bias term (line 155: `-(mean[0] * w_star).sum()` should be `+(mean[0] * w_star).sum()`), and the whole workaround is unnecessary.

## Fix

Move the `wstar_test_acc` computation to **before** the standardization block so it operates on raw `X_test` directly. This lets us drop the `if standardize` branch entirely.

### ~~Changes to `data/makeblobs.py`~~

- After `train_test_split` (line 135) and before `if standardize:` (line 137), insert:

```python
wstar_test_acc = None
wstar_file = dcfg.get('wstar_file')
if wstar_file is not None:
    w_star = np.load(wstar_file).astype(np.float32)
    z = X_test @ w_star
    preds = (z > 0).astype(np.int64)
    wstar_test_acc = float((preds == y_test).mean())
```

- Remove the old `wstar_test_acc` block (lines 149–161).

## Fix `wstar_test_acc` prediction sign — `data/makeblobs.py`

The existing (now-moved) block uses `(z > 0)` which casts `True → 1`. But z > 0 means the point is near +w_star, which is label **0**. This produces accuracy ≈ 1 − Φ(cscale) instead of Φ(cscale).

### Changes to `data/makeblobs.py`

Change:
```python
preds = (z > 0).astype(np.int64)
```
To:
```python
preds = (z <= 0).astype(np.int64)
```

## ~~Remove `wstar_test_acc` from per-step W&B logs~~

### Problem

`wstar_test_acc` is a constant (computed once at data load). Including it in every periodic diagnostics log produces a redundant flat line in W&B time-series charts.

### Changes to `methods/method_utils/diagnostics.py`

- Remove the per-step log of `wstar_test_acc` (inside `log_diagnostics`, within the `if self.wandb_loss_acc:` block):

```python
# remove these two lines:
if self.wandb_wstar_acc and self.wstar_test_acc is not None:
    log_data['wstar_test_acc'] = self.wstar_test_acc
```

The summary write at initialization (`wandb.summary['wstar_test_acc'] = self.wstar_test_acc`) is kept — it remains accessible as a run-level scalar without polluting charts.

## Log `bayes_accuracy` at each snapshot

`bayes_accuracy` is computed analytically (Φ(cscale)) and stored in the config, but is not currently logged per-snapshot. It provides a useful horizontal reference in time-series charts (unlike `wstar_test_acc`, which varies by run, `bayes_accuracy` is a fixed ceiling).

### Changes to `methods/method_utils/diagnostics_context.py`

Add field to `DiagnosticsRunContext`:
```python
bayes_accuracy: float | None = None
```

### Changes to `methods/SelectionMethod.py`

Pass the value when constructing the context:
```python
bayes_accuracy=self.config.get('bayes_accuracy'),
```

### Changes to `methods/method_utils/diagnostics.py`

In `__init__`, store it alongside `wstar_test_acc`:
```python
self.bayes_accuracy = context.bayes_accuracy
```

In `log_diagnostics`, inside the `if self.wandb_loss_acc:` block, add:
```python
if self.bayes_accuracy is not None:
    log_data['bayes_accuracy'] = self.bayes_accuracy
```
