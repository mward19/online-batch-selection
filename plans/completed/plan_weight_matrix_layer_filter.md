# Plan: Weight Matrix Diagnostics Layer Filtering

## Goal

Allow `wandb_weight_matrix_norms` to restrict logging to a subset of the network's weight matrices, either by specifying exact parameter names or by specifying the last n 2D weight matrices in the network.

## Changes

### ~~1. `methods/method_utils/weight_matrix.py`~~

- ~~Add `last_n_layers: int | None` parameter to `__init__`, stored as `self.last_n_layers`.~~
- ~~Remove the `param_names` kwarg from `log_metrics` and instead resolve the parameter filter entirely inside `log_metrics` based on `self.last_n_layers` and `self.param_names` (see below).~~
- ~~Add `param_names: list[str] | None` parameter to `__init__`, stored as `self.param_names`.~~
- ~~Resolution logic in `log_metrics`:~~
  1. ~~Collect all named parameters with `requires_grad=True` and 2D shape (these are the candidate weight matrices), preserving `model.named_parameters()` order.~~
  2. ~~If `self.param_names` is set, filter the full parameter list (not just 2D ones) to those names — `param_names` takes precedence and is not restricted to 2D params (the existing `_get_weight_info` already returns `None` for non-2D tensors, so non-matrix params are silently skipped).~~
  3. ~~Else if `self.last_n_layers` is set, take the last `self.last_n_layers` entries from the 2D candidates collected in step 1. If `self.last_n_layers` exceeds the number of 2D candidates, log a warning and use all candidates.~~
  4. ~~Else use all 2D candidates.~~

### ~~2. `methods/method_utils/diagnostics.py`~~

- ~~Read two new optional config keys in `__init__`:~~
  - ~~`weight_matrix_param_names` → `list` (default `None`)~~
  - ~~`weight_matrix_last_n_layers` → `int` (default `None`)~~
- ~~Pass both to `WeightMatrixDiagnostics(...)`.~~

### ~~3. Config (`configs/diagnostics/weight_matrix_tests.yaml`)~~

- ~~Add the active key and commented-out example for the other:~~
  ```yaml
  weight_matrix_last_n_layers: 5
  # weight_matrix_param_names: [layers.0.weight, layers.1.weight]
  ```
  {{The plan now sets `weight_matrix_last_n_layers: 5` as an active (uncommented) entry. The safe-fallback behavior (warn + use all layers) is captured in step 3 of the resolution logic above.}}
