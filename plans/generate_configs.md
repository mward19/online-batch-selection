Generate one basic config for each dataset. Each should use the train/val loss/acc diagnostics ("snapshots") and timing. Make the MNIST one also use the Progress diagnostic, and 10% label noise. 

---

## Plan (for review — nothing generated/implemented yet)

### Blocker: there is no `Timing` diagnostic yet
The registry (`methods/diagnostics/diagnostics.py`) has the snapshot leaves
`TrainLoss/TrainAcc/ValLoss/ValAcc` (+ `TrueLabel*`, `LogitNormL2`, `Progress`,
`ParamNorms/GradNorms/WeightMatrixNorms`, `LinearProbe`, `NTK`, `Checkpoint`) and
`SelectedPoints` (epoch-end). **Wall-clock timing was never built** (parity-audit
item D2.5). So "use timing" needs a small code change — a new leaf — which under
our workflow needs this plan approved first.

**Proposed `Timing` leaf (EPOCH_END).** The loop already tracks `self.total_time`
and `self.time_this_epoch`, freshly updated at the end of `train(epoch)`
(SelectionMethod.py:329-330) right before `after_epoch` → `run_epoch_end`. So:
- Add a `Timing(Diagnostic)` leaf in `diagnostics.py`, registered under
  `EPOCH_END_DIAGNOSTICS["Timing"]`, that logs
  `{"total_time": ..., "time_this_epoch": ...}` read from `shared_context`.
- Pass those two values into `run_epoch_end(...)` from `after_epoch`
  (SelectionMethod.py:200-205) — a 2-line addition next to `selected_mask`.
- Config usage: `Timing: {}` under `diagnostics.diagnostics`.
{{Epoch-end is the natural home (timing is per-epoch and fresh there). OK? Or do
you want it post-batch too? [[ ]]}}

### "snapshots" = the four loss/acc leaves
I'm reading "snapshots" as `TrainLoss + TrainAcc + ValLoss + ValAcc`. Confirm —
or do you also want the `TrueLabel*` pair (clean-label loss/acc) whenever a config
has label noise? [[ ]]

### Which datasets? (scope)
There are ~25 loader variants across 5 families
(`cifar`, `mnist`, `tinyimagenet`, `makeblobs`, `teacher_generated`). I read
"one basic config for each dataset" as **one per family**. Proposed picks +
baseline model/optim (modeled on the existing `cifar3_rholoss`/`makeblobs_uniform`
configs and each loader's required keys):

| family | dataset.name | model | optimizer | notes |
|---|---|---|---|---|
| makeblobs | `MakeBlobs` (or `MakeBlobs_Noise`) | `Linear` | SGD | synthetic; we already have `makeblobs_uniform.yaml` |
| mnist | `MNIST_Noise` | `LeNet` | SGD | **10% noise + Progress** per your ask |
| cifar | `CIFAR10` (or keep `CIFAR3`) | `ResNet` | SGD | |
| tinyimagenet | `TinyImageNet` | `ResNet` | SGD | heavy; smoke-test only on the cheap ones |
| teacher_generated | `Teacher_Generated` | `Linear`/`DeepLinear` | SGD | include this family at all? [[ ]] |

Open picks: [[CIFAR10 vs CIFAR3?]] · [[include teacher_generated?]] ·
[[MNIST vs FashionMNIST for the mnist one?]]

### Baseline method
"basic config" → I propose **`method: Full`** (train on everything, no selection)
as the cleanest baseline, since these are for observing train/val curves + timing,
not selection behavior. Alternative: `Uniform` with `ratio: 1.0`. Which? [[ ]]
(Note: `SelectedPoints` only makes sense with sub-selection + noise, so I'll
*omit* it from these basic configs unless you want it.)

### Diagnostics block (all configs)
```yaml
diagnostics:
  logging_defaults: { log_interval: logarithmic, save_init: 5, save_freq: 4 }
  diagnostics:
    TrainLoss: {}
    TrainAcc: {}
    ValLoss: {}
    ValAcc: {}
    Timing: {}            # new leaf (see blocker above)
    # MNIST config additionally:
    # Progress: {}
```
Plus `run_name_format` (the standard block) and a `wandb:` section in each.

### Open questions to resolve before I build
1. Approve the `Timing` leaf (epoch-end) code change? [[ ]]
2. Dataset scope: confirm the family picks above (CIFAR3 vs CIFAR10; FashionMNIST?;
   include teacher_generated?). [[ ]]
3. Baseline method: `Full` vs `Uniform@1.0`? [[ ]]
4. "snapshots" = just the 4 loss/acc leaves, or also `TrueLabel*` on noisy configs? [[ ]]
5. Where should these live — `configs/` (tracked) or `configs-temp/`? [[ ]]

### Checklist (after decisions)
- [ ] Implement `Timing` leaf + register + thread timing into `run_epoch_end`.
- [ ] Write one config per chosen dataset with the diagnostics block + run_name_format + wandb.
- [ ] MNIST config: `*_Noise`, `noise_percent: 0.1`, add `Progress`.
- [ ] GPU smoke-test the cheap ones (makeblobs, mnist) + commit.