# Plan: move diagnostics code out of `methods/method_utils/`, remove defunct scripts

Follow-up to the completed spring-cleaning refactor. Goal: relocate the
**diagnostics** code currently living in `methods/method_utils/` into the
`methods/diagnostics/` package (so each diagnostic's compute code lives with its
leaf), and delete scripts orphaned by the new `--config` CLI.

Per your note: the helpers that are **not strictly for diagnostics** stay in
`method_utils`. So `loss.py`, `optimizer.py`, and `build_teacher_model.py` remain
where they are (they're shared by `SelectionMethod`/`Optk`/`RhoLoss`/`Bayesian`),
and `method_utils/` is **not** deleted — it keeps those three. Only the
diagnostics modules move.

Branch: `spring-cleaning` (same as the refactor). Commit when done. No remote push.

## 1. Module relocations (diagnostics only)

| module | move to | why |
|---|---|---|
| `method_utils/ntk.py` (`NTKDiagnostics`) | inline into `methods/diagnostics/ntk.py` | only consumer is the `NTK` leaf |
| `method_utils/param_grad.py` (`ParamGradDiagnostics`) | inline into `methods/diagnostics/model_metrics.py` | only consumer is `ParamNorms`/`GradNorms` |
| `method_utils/probe.py` (`ProbeDiagnostics`) | inline into `methods/diagnostics/model_metrics.py` | only consumer is `LinearProbe` |
| `method_utils/weight_matrix.py` (`WeightMatrixDiagnostics`) | inline into `methods/diagnostics/model_metrics.py` | only consumer is `WeightMatrixNorms` |
| `method_utils/diagnostics_context.py` (`DiagnosticsRunContext`) | **deleted** | see §1.2 — managers own their context |

**Stay in `method_utils/` (not diagnostics):** `loss.py`, `optimizer.py`,
`build_teacher_model.py`, `__init__.py`. The package is kept; only its
`diagnostics_context` export is dropped from `__init__.py`.

### 1.2 Eliminate `DiagnosticsRunContext`; split static vs. dynamic on the manager
The fat frozen `DiagnosticsRunContext` (a holdover from the old
`DiagnosticsLogger`) goes away. The `DiagnosticsManager` instead holds **two**
dicts, separated by lifecycle:

- `self.static_context` — set **once** when `create_diagnostics` builds the
  manager and never changes: loaders, `save_dir`, `num_classes`,
  `num_train_samples`, `true_labels`, `noisy_indices`,
  `project_root`/`dataset_name`/`artifact_stem`/`seed`, the full `config`,
  `logger`, the `wstar/what/bayes` accuracies.
- `self.shared_context` — the existing **dynamic** per-step dict, updated each
  `run_diagnostics(...)`: `model`, `device`, `lr`, `checkpoint_state`,
  `selected_mask`.

`self.get_context()` returns the **merge** `{**self.static_context,
**self.shared_context}`, and **raises** if any key appears in both (the two key
sets must stay disjoint — no silent shadowing). This resolves the earlier "dict
vs. dataclass" question: both are plain dicts, distinguished only by when
they're written.

Leaves stop taking a `context` constructor arg. They read what they need from
`self.get_context()` at **run** time — e.g. `ForwardPass._run` reads
`ctx["fixed_train_loader"]`/`ctx["test_loader"]` by its `loader_key`;
`Checkpoint` reads `ctx["save_dir"]`; `NTK` reads
`ctx["num_classes"]`/`ctx["config"]`/`ctx["logger"]`; `PerSampleLossError` reads
`ctx["true_labels"]`; `SelectedPoints` reads `ctx["noisy_indices"]`.

Construction-time-only inputs that `create_diagnostics` itself holds (it receives
the resources from `SelectionMethod` as a **plain dict**, not a dataclass) are
used there directly: it builds each `LogSchedule` (needs `total_batches`,
`num_epochs`/`num_steps`, `save_init`/`save_freq`) and composes `log_path`
(`save_dir/logs/<name>.log`) before constructing the leaves, and seeds the
manager contexts. No 22-field object is threaded anywhere.

Note: this is a **diagnostics-framework refactor**, not just a file move — it
touches `base.py` (the manager), every leaf constructor/`__eq__`, and the engines
(`NTKDiagnostics` etc., which today read `context.<field>` at construction and
must instead receive explicit args / read `get_context()` at run time). Bigger
than the original "move method_utils" ask, but the right cleanup.

### 1.1 Inline (resolved)
Full inline: each diagnostic file holds its own engine + leaf(s).
`diagnostics/ntk.py` = `NTKDiagnostics` engine (~800 lines, fine) + `NTK` leaf;
`diagnostics/model_metrics.py` = `ParamGradDiagnostics`/`ProbeDiagnostics`/
`WeightMatrixDiagnostics` engines + the four leaves.

Import fixes inside the moved engines:
- `NTKDiagnostics` keeps `from methods.method_utils.build_teacher_model import build_teacher_model`
  (build_teacher_model stays in method_utils); its `__init__` takes explicit args
  (`fixed_train_loader`, `project_root`, `dataset_name`, `artifact_stem`, `seed`,
  `config`, `num_classes`, …) instead of a context, and the leaf builds it from
  `manager.static_context`.
- `ProbeDiagnostics` takes `train_loader`/`test_loader`/`logger` explicitly;
  `WeightMatrixDiagnostics` drops `context` (keeps `logger`).

## 2. Wiring changes (consumers + framework)

- `methods/SelectionMethod.py`: stop building a `DiagnosticsRunContext`. Build a
  plain **resources dict** (the static fields listed in §1.2) and pass it to
  `create_diagnostics(diagnostics_config, resources)`. `from .method_utils import *`
  still supplies `create_criterion`/`create_optimizer`/`create_scheduler`.
- `create_diagnostics.py`: signature `create_diagnostics(diagnostics_config, resources)`.
  For each manager: set `manager.static_context` from the static `resources`; build
  `LogSchedule`s and `log_path`s from `resources`; construct each leaf **without**
  a `context` arg (passing only its `params` and, for engine-backed leaves, the
  engine built from `resources`). Register leaves; return the runner.
- `methods/diagnostics/base.py`: `DiagnosticsManager` gains `self.static_context`
  (set once at build) alongside the existing `self.shared_context` (dynamic, per
  `run_diagnostics`); `get_context()` returns `{**static_context, **shared_context}`
  and **raises** on any key present in both.
- `methods/diagnostics/diagnostics.py` (the 5b leaves): drop the `context`
  constructor arg; read loaders/`save_dir`/`true_labels`/`noisy_indices`/
  `num_train_samples` from `self.get_context()` at run time. Update each `__eq__`
  accordingly (e.g. `ForwardPass` still dedups on `loader_key`).
- `methods/diagnostics/ntk.py`, `model_metrics.py`: inline the engines; the
  engines' constructors take explicit args (built by `create_diagnostics` from
  `resources`) instead of a `context`; the leaves read `model`/`device` from
  `self.get_context()` at run time. `NTKDiagnostics` keeps
  `from methods.method_utils.build_teacher_model import build_teacher_model`.
- `methods/method_utils/__init__.py`: drop the `DiagnosticsRunContext` export.
- `Optk.py` / `RhoLoss.py` / `Bayesian.py`: **no change**.
- Verify: grep that no `method_utils.{ntk,param_grad,probe,weight_matrix,diagnostics_context}`
  and no `DiagnosticsRunContext` references remain; import smoke test.

## 3. Scripts

**Delete all 5 `run_*.sh`** (they use the removed 5-flag CLI):
`run_cifar10.sh`, `run_cifar100.sh`, `run_mnist.sh`, `run_teacher_generated.sh`,
`run_teacher_generated_todd.sh`.

**Keep the SLURM submission script** `slurm_run_cifar_3_deep_linear.py` (the only
`slurm_*.py` left — the mnist/blobs ones were deleted earlier). It already submits
via `sbatch` (template + `generate_configs` + `--config` + `#SBATCH --requeue`).
- {{Rename per your suggestion: `slurm_run_cifar_3_deep_linear.py` →
  `run_cifar_3_deep_linear.py`, now that the `run_*.sh` are gone and the `run_`
  namespace is free. It stays the SLURM submission path (still emits `sbatch`
  jobs); I'll confirm it runs after the rename. I'll do this unless you object —
  say if you'd rather keep the `slurm_` prefix.}}

**Keep `perform_downloads.py` as-is** — it's used by the SLURM submission flow.
{{No good reason to move it; it's a top-level entry point like `main.py`, so it
stays at repo root.}}

Other kept (current): `main.py`, `run_dir.py`, `create_diagnostics.py`,
`generate_configs.py`, `utils.py`, `wandb-sync-daemon.py`.

## 4. Verify

- Import smoke test under the env python.
- GPU smoke test: `srun … main.py --config configs-temp/makeblobs_smoke.yaml --wandb_not_upload`
  (the config that exercises every diagnostic leaf, incl. NTK/probe/param/weight).

## 5. Checklist
- [x] ~~`base.py`: add `DiagnosticsManager.static_context` (set once at build)
      beside `shared_context` (dynamic); `get_context()` returns the merge.~~ (raises on key collision)
- [x] ~~Delete `DiagnosticsRunContext`; `SelectionMethod` builds a resources dict;
      `create_diagnostics(diagnostics_config, resources)` sets each manager's
      `static_context`, builds schedules/log_paths, constructs leaves without a
      `context` arg.~~
- [x] ~~Drop the `context` arg from the 5b leaves; read resources from
      `get_context()` at run time; fix each `__eq__`.~~
- [x] ~~Inline `ntk.py` → `methods/diagnostics/ntk.py` and
      `param_grad/probe/weight_matrix` → `methods/diagnostics/model_metrics.py`
      (full inline); engines take explicit args, leaves read per-step values from
      `get_context()`.~~
- [x] ~~Delete the 4 moved diagnostics modules + `diagnostics_context.py` from
      `method_utils/`; drop the export from `__init__.py`. Keep
      `loss.py`/`optimizer.py`/`build_teacher_model.py`.~~
- [x] ~~Delete the 5 `run_*.sh`; rename `slurm_run_cifar_3_deep_linear.py` →
      `run_cifar_3_deep_linear.py`.~~
- [x] ~~Import smoke test + GPU smoke test.~~ (makeblobs, all leaves logged incl. NTK/probe/param/weight; checkpoint under snapshots/)
- [ ] Commit on `spring-cleaning`; move this plan to `plans/completed/`.
