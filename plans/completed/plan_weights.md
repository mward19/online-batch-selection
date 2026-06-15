# Plan: WeightMatrixDiagnostics

Monitor per-layer weight matrix norms throughout training and log them to W&B. Primarily motivated by deep linear network experiments where matrix norms (Frobenius, spectral) are scientifically meaningful quantities.

---

## What Already Exists

`weights.py` has a partial implementation:
- `_get_weight_info(name, p)` computes Frobenius norm, matrix 2-norm (largest singular value), and an "alignment metric" (σ_max / ‖W‖_F).
- `log_metrics(model, param_names=None)` iterates named parameters and calls `_get_weight_info`.

`diagnostics.py` already imports `WeightDiagnostics` and constructs it (line 86–90), and has a placeholder `log_data.update()` on line 182 where the metrics should be inserted.

---

## Bugs to Fix

### ~~1. `__init__` missing `enabled` parameter~~
~~`diagnostics.py` calls `WeightDiagnostics(logger=..., context=..., enabled=True)` but the constructor only accepts `logger` and `context`. This crashes at startup. The `enabled` parameter needs to be added to `__init__` and stored as `self.enabled`.~~

### ~~2. Broken W&B logging structure~~
~~`log_metrics` currently does:~~
```python
log_data[f'diagnostics/param_info/{name}'] = self._get_weight_info(name, p)
```
~~This assigns a **dict as a value**, which `wandb.log` will not handle correctly — it expects scalar/tensor/wandb-type values. Fix: flatten the inner dict's keys into the outer dict:~~
```python
log_data[f'diagnostics/weight_norms/{name}/frobenius'] = ...
log_data[f'diagnostics/weight_norms/{name}/spectral'] = ...
log_data[f'diagnostics/weight_norms/{name}/alignment'] = ...
```
~~`_get_weight_info` should return `None` for non-2D parameters (see filtering decision below), and `log_metrics` should skip `None` returns instead of trying to insert them.~~

### ~~3. Missing `.detach().cpu().item()`~~
~~Norms are computed on live GPU tensors without detaching from the autograd graph. This holds the graph alive unnecessarily and keeps the values as tensors rather than Python scalars. All norm values should have `.detach().cpu().item()` applied before being put into `log_data`.~~

### ~~4. `log_data.update()` is a no-op~~
~~Line 182 of `diagnostics.py` is `log_data.update()` with no argument — a placeholder. It should be:~~
```python
if self.weight_matrix_diagnostics.enabled:
    log_data.update(self.weight_matrix_diagnostics.log_metrics(model))
```

---

## Design Decisions

### Which norms to compute
Keep all three quantities computed by the current code:
- **Frobenius norm** ‖W‖_F: overall scale of the weight matrix.
- **Spectral norm** (matrix 2-norm, σ_max): largest singular value, computed via `torch.linalg.matrix_norm(p, ord=2)`. This triggers an SVD internally.
- **Alignment metric** σ_max / ‖W‖_F: ranges from 1/√min(m,n) (uniform singular value spectrum) to 1 (rank-1 matrix). Measures how "spiked" the singular value distribution is. Rename the key to `alignment` (from "Alignment Metric") for clarity in W&B.

**SVD cost**: `torch.linalg.matrix_norm(p, ord=2)` computes a full SVD. For the deep linear networks in this project (hidden dims of 3–100), this is negligible. For larger models (e.g., ResNet), this could be expensive — but since `WeightMatrixDiagnostics` is only called at log intervals (not every batch), this is acceptable.

### Filtering non-matrix parameters
Skip non-2D parameters **silently** (no warning). Biases, batch-norm parameters, etc. have no meaningful matrix norm. `_get_weight_info` returns `None` for `len(shape) != 2`; `log_metrics` skips `None` returns without logging anything. The existing warning is removed entirely — it would fire for every bias at every log step, spamming the console.

**Rename**: Since this class only handles matrix (2D) parameters, rename for clarity:
- ~~File: `weights.py` → `weight_matrix.py`~~
- ~~Class: `WeightDiagnostics` → `WeightMatrixDiagnostics`~~

~~Update the import in `diagnostics.py` accordingly.~~

### Config-driven enable/disable
~~Replace the hardcoded `enabled=True` in `diagnostics.py` with a config key. Add `wandb_weight_matrix_norms` to the diagnostics YAML.~~ This is a **true/false flag** — it controls whether weight matrix norm logging is enabled at all. It is not a list of parameter names (filtering by name is handled separately via the optional `param_names` kwarg, which is not wired to config for now).

~~In `DiagnosticsLogger.__init__`:~~
```python
self.wandb_weight_matrix_norms = bool(diagnostics_config.get('wandb_weight_matrix_norms', False))
```
~~Pass this to `WeightMatrixDiagnostics(enabled=self.wandb_weight_matrix_norms, ...)`.~~

### `param_names` filtering
~~`log_metrics` already accepts an optional `param_names` list. Do not wire this to config — leave it as an optional kwarg for future use. The default (all 2D parameters) is the correct behavior for this project. Add a brief comment in the code noting that `param_names` is intentionally not config-driven.~~

### `context` parameter
~~Keep `context: DiagnosticsRunContext` in the constructor for consistency with `NTKDiagnostics` and `ProbeDiagnostics`, and because it may be used later. Store it as `self.context`. The call site in `diagnostics.py` already passes `context=context`.~~

### W&B key naming convention
~~Use: `diagnostics/weight_norms/{param_name}/frobenius`, `.../spectral`, `.../alignment`~~

~~This is consistent with the `diagnostics/` prefix used elsewhere and groups per-layer metrics cleanly in the W&B UI.~~

---

## Summary of Changes

### `weight_matrix.py` (renamed from `weights.py`)
1. ~~Rename file to `weight_matrix.py`; rename class to `WeightMatrixDiagnostics`.~~
2. ~~Add `enabled` parameter to `__init__`; store as `self.enabled`. Keep `context` parameter; store as `self.context`.~~
3. ~~In `_get_weight_info`:~~
   - ~~Remove the `logger.info` warning for non-2D params; just `return None` silently.~~
   - ~~Apply `.detach().cpu().item()` to all three norm values.~~
   - ~~Rename keys to `frobenius`, `spectral`, `alignment`.~~
4. ~~In `log_metrics`:~~
   - ~~Add early return `{}` if `not self.enabled`.~~
   - ~~Flatten the inner dict into `log_data` using `diagnostics/weight_norms/{name}/{metric}` keys; skip parameters where `_get_weight_info` returns `None`.~~
   - ~~Add a brief comment that `param_names` is intentionally not config-driven.~~

### `diagnostics.py`
1. ~~Update import: `from methods.method_utils.weight_matrix import WeightMatrixDiagnostics`.~~
2. ~~Read `wandb_weight_matrix_norms` from config and store on `self` (default `False`).~~
3. ~~Update constructor call: `WeightMatrixDiagnostics(logger=self.logger, context=context, enabled=self.wandb_weight_matrix_norms)`.~~
4. ~~Replace the no-op `log_data.update()` on line 182 with:~~
   ```python
   if self.weight_matrix_diagnostics.enabled:
       log_data.update(self.weight_matrix_diagnostics.log_metrics(model))
   ```

### Diagnostics YAML configs
~~Add `wandb_weight_matrix_norms: true` to `configs/diagnostics/all_log_interval.yaml`. The default remains `false` (omit from `snapshots_log_interval.yaml`).~~

---

## Open Questions
- Should we also log **effective rank** (`exp(H)` where `H` is the entropy of the normalized squared singular values)? This is more expensive (full SVD is already happening, so it's just extra arithmetic on the singular values). Worth considering for a follow-up, not this plan.
