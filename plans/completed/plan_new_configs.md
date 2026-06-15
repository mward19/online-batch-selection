# Plan: Dimensionality sweep configs

## Decisions

- **Dims**: `[32, 1024, 8192, 32768]`
- **`hidden_dim`**: fixed at 1024 (reuse `deep_linear_1024_3layer.yaml` for all dims)
- **Config generation**: slurm script patches committed template files and writes to a gitignored `generated/` directory — no committed configs per (dim, cscale)

---

## ~~Template files (committed to the repo)~~

**`configs/makeblobs/method/rholoss-0.1-hyperplane-template.yaml`** — same as current `rholoss-0.1-hyperplane.yaml` but with:
```yaml
teacher_model_path: null
```

**`configs/makeblobs/method/bayesian-0.1-hyperplane-template.yaml`** — same as current `bayesian-0.1-hyperplane.yaml` but with:
```yaml
teacher_model_path: null
```

**`configs/makeblobs/data/makeblobs-template.yaml`** — based on `makeblobs_1024d_2class.yaml` but with:
```yaml
dataset:
  n_features: null
  input_dim: null
  center_file: null
  # all other keys kept as-is
```

---

## ~~Generated directory layout~~

```
configs/makeblobs/generated/   ← gitignored
  method/
    rholoss-0.1-hyperplane_d{dim}_cscale{cscale}.yaml
    bayesian-0.1-hyperplane_d{dim}_cscale{cscale}.yaml
  data/
    makeblobs_d{dim}_cscale{cscale}.yaml
```

Model config is fixed: `configs/makeblobs/model/deep_linear_saxe/deep_linear_1024_3layer.yaml`.

---

## ~~Cleanup~~

Delete superseded method configs:
- `configs/makeblobs/method/rholoss-0.1-hyperplane-cscale{0.1,0.5,1.0}.yaml`
- `configs/makeblobs/method/bayesian-0.1-hyperplane-cscale{0.1,0.5,1.0}.yaml`
- `configs/makeblobs/method/rholoss-0.1-hyperplane.yaml`
- `configs/makeblobs/method/bayesian-0.1-hyperplane.yaml`
- `configs/makeblobs/method/bayesian-0.1.yaml`

Add `configs/makeblobs/generated/` to `.gitignore`.

---

## ~~`slurm_run_blobs_deep_linear.py` changes~~

Add imports: `import yaml`, `import copy`

{{4. Kept. They don't have per-(dim, cscale) method configs, but they still run against each generated data config as baselines. Added as `METHODS_FIXED`; see jobs construction below.}}

Replace the lists block with:
```python
DIMS          = [32, 1024, 8192, 32768]
CENTER_SCALES = [0.1, 0.5, 1.0]
METHOD_NAMES  = ["rholoss-0.1-hyperplane", "bayesian-0.1-hyperplane"]
METHODS_FIXED = [
    f"{CONFIG_DIR}/method/uniform-0.1.yaml",
    f"{CONFIG_DIR}/method/divbs-0.1.yaml",
]
MODEL_CONFIG  = f"{CONFIG_DIR}/model/deep_linear_saxe/deep_linear_1024_3layer.yaml"
GEN_DIR       = Path(CONFIG_DIR) / "generated"
```

Remove `DATAS`, `MODELS`, and `METHODS`.

Add generation helper (place before the pregeneration loop):
```python
def write_generated_configs(dims, center_scales):
    rholoss_tmpl  = yaml.safe_load(open(f"{CONFIG_DIR}/method/rholoss-0.1-hyperplane-template.yaml"))
    bayesian_tmpl = yaml.safe_load(open(f"{CONFIG_DIR}/method/bayesian-0.1-hyperplane-template.yaml"))
    data_tmpl     = yaml.safe_load(open(f"{CONFIG_DIR}/data/makeblobs-template.yaml"))

    for d, cs in product(dims, center_scales):
        teacher_path = f"models/teacher/makeblobs_{d}d_cscale{cs}_hyperplane_alpha1.0_nseed0.pth"

        for name, tmpl in [
            (f"rholoss-0.1-hyperplane_d{d}_cscale{cs}",  rholoss_tmpl),
            (f"bayesian-0.1-hyperplane_d{d}_cscale{cs}", bayesian_tmpl),
        ]:
            cfg = copy.deepcopy(tmpl)
            cfg['teacher_model_path'] = teacher_path
            p = GEN_DIR / "method" / f"{name}.yaml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(yaml.dump(cfg))

        cfg = copy.deepcopy(data_tmpl)
        cfg['dataset']['n_features']  = d
        cfg['dataset']['input_dim']   = [1, d]
        cfg['dataset']['center_file'] = f"models/teacher/makeblobs_{d}d_cscale{cs}_centers_seed42.npy"
        p = GEN_DIR / "data" / f"makeblobs_d{d}_cscale{cs}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.dump(cfg))
```

Replace pregeneration loop:
```python
for dim, cscale in product(DIMS, CENTER_SCALES):
    print(f"Generating teacher for dim={dim}, center_scale={cscale}...")
    subprocess.run([
        "python", "data/make_blobs_teacher.py",
        "--n_features", str(dim), "--center_scale", str(cscale),
        "--center_seed", "42", "--alpha", "1.0",
        "--noise_seed", "0", "--out_dir", "models/teacher",
    ], check=True)

write_generated_configs(DIMS, CENTER_SCALES)
```

Replace jobs construction:
```python
jobs = (
    [
        (
            seed,
            str(GEN_DIR / "data"   / f"makeblobs_d{dim}_cscale{cscale}.yaml"),
            MODEL_CONFIG,
            optim,
            str(GEN_DIR / "method" / f"{method_name}_d{dim}_cscale{cscale}.yaml"),
        )
        for seed, dim, cscale, optim, method_name
        in product(SEEDS, DIMS, CENTER_SCALES, OPTIMS, METHOD_NAMES)
    ] + [
        (
            seed,
            str(GEN_DIR / "data" / f"makeblobs_d{dim}_cscale{cscale}.yaml"),
            MODEL_CONFIG,
            optim,
            method,
        )
        for seed, dim, cscale, optim, method
        in product(SEEDS, DIMS, CENTER_SCALES, OPTIMS, METHODS_FIXED)
    ]
)
```
