# Plan: Option D — Empirical w_star test accuracy logged to W&B

## Goal

Load `w_star` from the saved `.npy` file, evaluate the Bayes-optimal linear classifier on the
(standardized) test set once at the start of training, and log `wstar_test_acc` to W&B — both
as a `wandb.summary` entry (visible in the runs table) and as a constant alongside `val_acc`
at every logging step (so it appears as a reference line on the accuracy plot).

This is independent of Option A: Option A logs the *analytic* `bayes_accuracy` (= Φ(cscale))
as a W&B config key. Option D logs the *empirical* accuracy of the exact `w_star` rule on the
finite standardized test set as a metric.

---

## Standardization correction

The data loader z-scores features using per-feature train statistics `(train_mean, train_std)`.
`w_star` is a unit vector in the *original* feature space. To apply the `w_star` decision rule
on the standardized test data `X_std = (X - train_mean) / train_std`, note that:

```
x · w_star > 0
⟺ (x_std * train_std + train_mean) · w_star > 0
⟺ x_std · w_star_eff > bias
```

where `w_star_eff = train_std * w_star` (elementwise) and `bias = -train_mean · w_star`.

Since both classes are balanced and centered at `±cscale * w_star`, the empirical `train_mean`
is close to zero and `train_std ≈ 1` per feature, so the correction is small. But the plan
uses the exact formula to be precise.

---

## Files to change

### ~~1. `slurm_run_blobs_deep_linear.py` — inject `wstar_file` into generated data config~~

In `write_generated_configs`, after the existing `center_file` line, add:

```python
cfg['dataset']['wstar_file'] = f"models/teacher/makeblobs_{d}d_cscale{cs}_wstar_seed42.npy"
```

This makes the wstar path available to the data loader at training time via `config['dataset']`.

### ~~2. `data/makeblobs.py` — compute `wstar_test_acc` in the loader~~

After the standardization block (lines ~135–140) and before building `wrapped_dataset`, add a
block in `MakeBlobs` (and `MakeBlobs_Noise` if used for blobs experiments — check) that:

1. Reads `wstar_file = dcfg.get('wstar_file')`.
2. If set, loads `w_star = np.load(wstar_file).astype(np.float32)`.
3. Applies the exact standardization correction:
   ```python
   w_eff = train_std[0] * w_star          # elementwise; train_std shape (1, d) from keepdims
   bias  = -(train_mean[0] * w_star).sum()  # scalar
   z     = X_test @ w_eff + bias           # shape (n_test,)
   preds = (z > 0).astype(np.int64)       # class 0 if positive
   wstar_test_acc = float((preds == y_test).mean())
   ```
   (When `standardize=False`, just use `z = X_test @ w_star`, `bias = 0`.)
4. Adds `wstar_test_acc` to the payload dict in `_makeblobs_output`.

`_makeblobs_output` already receives `train_dset`/`test_dset` by value; add an optional
`wstar_test_acc: float | None = None` parameter and include it in the returned dict when
not None.

### ~~3. `methods/method_utils/diagnostics_context.py` — add `wstar_test_acc` field~~

Add an optional field to the frozen dataclass:

```python
wstar_test_acc: float | None = None
```

### ~~4. `methods/SelectionMethod.py` — thread `wstar_test_acc` into the context~~

In `__init__`, where `DiagnosticsRunContext` is constructed, add:

```python
wstar_test_acc=self.data_info.get('wstar_test_acc'),
```

### ~~5. `methods/method_utils/diagnostics.py` — log the value~~

In `DiagnosticsLogger.__init__`:

- Read the new flag: `self.wandb_wstar_acc = bool(diagnostics_config.get('wandb_wstar_acc', False))`
- Store the value: `self.wstar_test_acc = context.wstar_test_acc`
- Log once to W&B summary (so it appears in the runs table):
  ```python
  if self.wandb_wstar_acc and self.wstar_test_acc is not None:
      wandb.summary['wstar_test_acc'] = self.wstar_test_acc
  ```

In `DiagnosticsLogger.log_diagnostics`, inside the `if self.wandb_loss_acc:` block, add:

```python
if self.wstar_test_acc is not None and self.wandb_wstar_acc:
    log_data['wstar_test_acc'] = self.wstar_test_acc
```

This emits the constant on every logging step so W&B can render it as a horizontal reference
line on the `val_acc` vs. step chart.

### ~~6. `configs/diagnostics/weight_matrix_tests.yaml` — enable the flag~~

Add one line:

```yaml
wandb_wstar_acc: true
```

---

## What does NOT change

- `main.py` — no changes needed; config is already merged and passed through.
- `configs/makeblobs/data/makeblobs-template.yaml` — `wstar_file` is injected at generation
  time (step 1), not baked into the template, so runs without a wstar file continue to work.
- Any method files — this is purely a data-loader + diagnostics concern.

---

## Scope

Only `MakeBlobs` (and `MakeBlobs_Noise` if applicable) returns `wstar_test_acc`. For all
other datasets the key is absent from `data_info`, the context field stays `None`, and
`DiagnosticsLogger` simply skips logging it. No guard needed in the slurm script.
