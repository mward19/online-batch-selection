# Plan: Iterate over model config paths in `slurm_run_blobs_deep_linear.py`

## Changes to `slurm_run_blobs_deep_linear.py`

Replace:
```python
MODEL_LAYERS = [3, 16]
MODEL_CONFIG  = f"{CONFIG_DIR}/model/deep_linear_saxe/deep_linear_1024_3layer.yaml"
```

With:
```python
MODEL_CONFIGS = [
    f"{CONFIG_DIR}/model/deep_linear_saxe/deep_linear_1024_3layer.yaml",
    f"{CONFIG_DIR}/model/deep_linear_saxe/deep_linear_1024_16layer.yaml",
]
```

In the jobs construction, replace the two hardcoded `MODEL_CONFIG` references with `model` and add `MODEL_CONFIGS` to each product:

```python
jobs = (
    [
        (
            seed,
            str(GEN_DIR / "data"   / f"makeblobs_d{dim}_cscale{cscale}.yaml"),
            model,
            optim,
            str(GEN_DIR / "method" / f"{method_name}_d{dim}_cscale{cscale}.yaml"),
        )
        for seed, dim, cscale, optim, method_name, model
        in product(SEEDS, DIMS, CENTER_SCALES, OPTIMS, METHODS_HYPERPLANE, MODEL_CONFIGS)
    ] + [
        (
            seed,
            str(GEN_DIR / "data" / f"makeblobs_d{dim}_cscale{cscale}.yaml"),
            model,
            optim,
            method,
        )
        for seed, dim, cscale, optim, method, model
        in product(SEEDS, DIMS, CENTER_SCALES, OPTIMS, METHODS_FIXED, MODEL_CONFIGS)
    ]
)
```

The `model_id` regex (`re.search(r'deep_linear_(.+)\.yaml', model)`) already handles both filenames correctly — no change needed there.
