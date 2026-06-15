# Plan: MakeBlobs 1024-D Deep Linear Experiments

Test online batch selection methods on 2-class, 1024-dimensional synthetic Gaussian blob data using a 3-hidden-layer DeepLinear network. RhoLoss requires a teacher model constructed from the ground-truth separating hyperplane.

---

## Overview of New Files

| File | Type |
|---|---|
| `configs/makeblobs/data/makeblobs_1024d_2class.yaml` | data config |
| `configs/makeblobs/model/deep_linear_saxe/deep_linear_1024_3layer.yaml` | model config |
| `configs/makeblobs/method/rholoss-0.1-hyperplane.yaml` | method config (RhoLoss only) |
| `data/make_blobs_teacher.py` | geometry + teacher generation script (new) |
| `bayes_error_derivation.md` | derivation of Bayes error and teacher noise analysis |

Existing `configs/makeblobs/method/{bayesian,divbs,uniform,full}-*.yaml` configs are reused as-is.

---

## Data Config: 2-Class 1024-D Blobs

### Design

The two cluster centers are symmetric around the origin: `c1 = -c0`. The separating direction `w*` is a random unit vector — more realistic than `e_1` and avoids accidental alignment with the network's initialization.

**Single source of truth for `w*`**: `data/make_blobs_teacher.py` generates `w*`, saves it as a `.npy` file, and also saves the noised teacher. The data loader reads the centers from that same `.npy` file via `centers_type: from_file`. This avoids duplicating RNG logic across scripts.

With `center_scale = 1.0` (unit-norm centers) and `cluster_std = 1.0`, the blobs are **2σ apart** along the separating direction, giving Bayes error ≈ 15.9% — enough overlap for the selection methods to have something interesting to do. See `bayes_error_derivation.md` for the full derivation; the general formula is:

$$P(\text{Bayes error}) = \Phi\!\left(-\frac{\texttt{center\_scale}}{\texttt{cluster\_std}}\right)$$

### Required code changes: `data/makeblobs.py`

Add a `centers_type: from_file` option. When present, centers are loaded from a `.npy` file (shape `(2, n_features)`) rather than generated. This applies to both `MakeBlobs` and `MakeBlobs_Noise`.

```python
centers_type = dcfg.get('centers_type', None)
if centers_type == 'from_file':
    center_file = dcfg['center_file']
    centers = np.load(center_file)  # shape (2, n_features)
    if centers.shape != (2, n_features):
        raise ValueError(f'center_file has shape {centers.shape}, expected (2, {n_features})')
    num_classes = 2
elif isinstance(centers, int):
    num_classes = centers
else:
    num_classes = len(centers)
```

This replaces/extends the existing `centers = dcfg.get('centers', 3)` logic. The `num_classes` assignment moves into the same conditional block.

### `configs/makeblobs/data/makeblobs_1024d_2class.yaml`

```yaml
dataset:
  name: MakeBlobs
  generate_new: False
  n_samples: 10000
  n_features: 1024
  centers_type: from_file
  center_file: models/teacher/makeblobs_1024d_centers_seed42.npy
  cluster_std: 1.0
  test_size: 0.2
  standardize: True
  num_classes: 2
  input_dim: [1, 1024]
  include_teacher: False
```

`n_samples = 10000` is changeable in the config and is fine for a synthetic linear problem. `cluster_std = 1.0` with unit-norm centers gives 2σ separation and modest class overlap.

---

## Model Config

### `configs/makeblobs/model/deep_linear_saxe/deep_linear_1024_3layer.yaml`

```yaml
networks:
  type: DeepLinear
  params:
    m_type: deeplinear
    hidden_dim: 1024
    num_hidden_layers: 3
```

Architecture: input (1024) → Linear(1024→1024) × 3 → classifier (1024→2). No nonlinearities.

---

## Teacher Model for RhoLoss

### Concept

The ground-truth separating hyperplane has normal `w*`. The teacher uses a noised version:

$$\tilde{w} = w^* + \frac{\alpha}{\sqrt{d}}\,\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I_{1024}), \qquad \hat{w} = \tilde{w}/\|\tilde{w}\|$$

The teacher classifies as a pure inner product: `logits = [-w_hat·x, +w_hat·x]`.

### Noise parameterization

Use the dimensionless parameter `alpha = noise_std * sqrt(d)` so that `alpha=1` means equal signal and noise magnitude, giving `θ = 45°` between teacher and `w*`. See `bayes_error_derivation.md` for the full analysis.

| alpha | noise\_std (d=1024) | θ | teacher quality |
|---|---|---|---|
| 0.5 | 0.016 | 27° | mildly corrupted |
| 1.0 | 0.031 | 45° | half-and-half |
| 2.0 | 0.063 | 63° | heavily corrupted |

{{`noise_std` is a CLI argument to `make_blobs_teacher.py` (an offline step). Different noise levels produce different `.pth` files. Default: `alpha = 0.5` → `noise_std = 0.016`, giving θ ≈ 27°.}}

**Sweeping alpha**: create one method config per alpha value (e.g., `rholoss-0.1-hyperplane-alpha0.5.yaml`, `alpha0.5`, `alpha2.0`), each pointing to its own `.pth` file. Run `make_blobs_teacher.py` once per alpha. Add all three to the `METHODS` list in a future SLURM sweep script — no new machinery required.

### Teacher architecture: `Linear` (simple inner product)

The teacher is a `Linear` model (`models/Linear.py`): a single `nn.Linear(1024, 2)` with no hidden layers. Its state dict is:
- `fc.weight = [-w_hat; +w_hat]` (shape 2×1024)
- `fc.bias = zeros(2)`

**Required code change: `methods/method_utils/build_teacher_model.py`**

Currently, `local_pretrained` reads the model architecture from `config['networks']` (the student). Fix: when `config['local_pretrained']` has a `type` key, use it; otherwise fall back to `config['networks']`:

```python
if teacher_model_source == 'local_pretrained':
    logger.info(f'Loading teacher model from {teacher_model_path}')
    lp_config = config.get('local_pretrained', {})
    if lp_config.get('type'):
        model_type = lp_config['type']
        model_args = (lp_config.get('params') or {}) | config['dataset']
    else:
        model_type = config['networks']['type']
        model_args = config['networks']['params'] | config['dataset']
    model = getattr(models, model_type)(**model_args)
    model.load_state_dict(torch.load(teacher_model_path, map_location='cpu'))
    return model
```

{{`local_pretrained` is the right key to reuse — adding a new `teacher_model_source` value (e.g., `linear_pretrained`) would be cleaner semantically but requires a new code path. The minimal change of checking `local_pretrained.type` is sufficient.}} [[Yes, I suppose it is technically a pretrained model, even if the training is pretty silly.]]

### `make_blobs_teacher.py` (new script)

Single script that generates all geometry artifacts. Must be run once before training.

```
python make_blobs_teacher.py \
    --n_features 1024 \
    --center_scale 1.0 \
    --center_seed 42 \
    --alpha 0.5 \
    --noise_seed 0 \
    --out_dir models/teacher
```

This saves four files to `out_dir`:
1. `makeblobs_1024d_centers_seed42.npy` — shape `(2, 1024)`, the actual cluster centers `[+center_scale * w*, -center_scale * w*]`. Read by the data loader.
2. `makeblobs_1024d_wstar_seed42.npy` — shape `(1024,)`, the ground-truth unit direction `w*`. Saved for reference and analysis.
3. `makeblobs_1024d_wnoised_alpha0.5_nseed0.npy` — shape `(1024,)`, the noised unit direction `w_hat`. Saved for reference.
4. `makeblobs_1024d_hyperplane_alpha0.5_nseed0.pth` — state dict of the `Linear` teacher model.

Logic:
1. Generate `w*`: sample `rng = np.random.default_rng(center_seed)`, draw and normalize
2. Build centers array; save as `centers_seed{center_seed}.npy`
3. Save `w*` as `wstar_seed{center_seed}.npy`
4. Sample noise `ε ~ N(0, I)` using `noise_seed`; compute `w_hat = normalize(w* + (alpha/sqrt(d)) * ε)`
5. Save `w_hat` as `wnoised_alpha{alpha}_nseed{noise_seed}.npy`
6. Build state dict: `fc.weight = stack([-w_hat, +w_hat])`, `fc.bias = zeros(2)`; save as `.pth`

### `configs/makeblobs/method/rholoss-0.1-hyperplane.yaml`

```yaml
method: RhoLoss
method_opt:
  ratio: 0.1
  balance: False
  ratio_scheduler: constant
  warmup_epochs: 0
  uniform_epochs: 0

teacher_model_source: local_pretrained
teacher_model_path: models/teacher/makeblobs_1024d_hyperplane_alpha0.5_nseed0.pth

local_pretrained:
  type: Linear
  params:
    m_type: linear
```

The `local_pretrained.type: Linear` key is what `build_teacher_model.py` reads to select the `Linear` architecture instead of the student's `DeepLinear`.

---

## Workflow

1. **Create output directory** (once):
   ```bash
   mkdir -p models/teacher
   ```
2. **Generate geometry + teacher** (once per alpha value):
   ```bash
   python data/make_blobs_teacher.py \
       --n_features 1024 --center_scale 1.0 --center_seed 42 \
       --alpha 0.5 --noise_seed 0 --out_dir models/teacher
   ```
3. **Cache dataset labels** (once):
   ```bash
   python save_labels.py --data configs/makeblobs/data/makeblobs_1024d_2class.yaml
   ```
4. **Run a single experiment**:
   ```bash
   python main.py \
     --method configs/makeblobs/method/rholoss-0.1-hyperplane.yaml \
     --data   configs/makeblobs/data/makeblobs_1024d_2class.yaml \
     --model  configs/makeblobs/model/deep_linear_saxe/deep_linear_1024_3layer.yaml \
     --optim  configs/makeblobs/optim/adamw-320-0.001-0.01.yaml \
     --wandb_not_upload
   ```
   Other methods substitute their own `--method` config.

---

## Summary of Changes

### New files
- [x] ~~`configs/makeblobs/data/makeblobs_1024d_2class.yaml`~~
- [x] ~~`configs/makeblobs/model/deep_linear_saxe/deep_linear_1024_3layer.yaml`~~
- [x] ~~`configs/makeblobs/method/rholoss-0.1-hyperplane.yaml`~~
- [x] ~~`data/make_blobs_teacher.py`~~

### Code changes
- [x] ~~`data/makeblobs.py`: add `centers_type: from_file` / `center_file` handling (both `MakeBlobs` and `MakeBlobs_Noise`)~~
- [x] ~~`methods/method_utils/build_teacher_model.py`: read teacher architecture from `config['local_pretrained']['type']` when present, falling back to `config['networks']`~~

---

## Open Questions

1. **Teacher noise level**: Default `alpha = 0.5` (θ = 27°). Sweep alpha ∈ {0.5, 1.0, 2.0} via separate method configs + a future SLURM sweep script.
2. **Optim config**: Reusing `configs/makeblobs/optim/adamw-320-0.001-0.01.yaml` for now.
3. **SLURM sweep**: Not yet — minimum example first.
