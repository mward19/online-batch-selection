# Plan: Add `USE_SLURM` Toggle to `slurm_run_*.py`

Add a `USE_SLURM = True` flag near the top of each script. When `False`, jobs run serially in the current process instead of being submitted to SLURM via `sbatch`. The sync daemon runs either way.

---

## Changes per script

The pattern is identical across all three scripts. For each:

### 1. Add flag

```python
USE_SLURM = True
```

Placed immediately after the `WANDB_PROJECT` line.

### 2. Replace the `sbatch` submission block

**Before:**
```python
sbatch_script = dedent(...)
subprocess.run(["sbatch"], input=sbatch_script, text=True, check=True)
```

**After:**
```python
python_cmd = [
    "python", "main.py",
    "--method", method,
    "--data", data,
    "--model", model,
    "--optim", optim,
    "--diagnostics", DIAGNOSTICS,
    "--seed", str(seed),
    "--save_dir", save_dir,
    "--wandb_project", WANDB_PROJECT,
]

if USE_SLURM:
    sbatch_script = dedent(
        f"""\
        #!/bin/bash
        ...

        {' '.join(python_cmd + ['--wandb_not_upload'])} \\
        """
    )
    subprocess.run(["sbatch"], input=sbatch_script, text=True, check=True)
else:
    subprocess.run(python_cmd, check=True)
```

{{1. The `python_cmd` list is built without `--wandb_not_upload`. The SLURM path appends it inline in the sbatch script body (as a shell string); the local path runs `python_cmd` directly, letting W&B upload live.}}

### 3. Update tqdm description

```python
desc="Submitting jobs" if USE_SLURM else "Running jobs"
```

### 4. Update final print + sync daemon

When `USE_SLURM = True`: print "All jobs submitted. Running Weights & Biases sync daemon. Ctrl+C to stop syncing" and run the sync daemon as before.

When `USE_SLURM = False`: print "All jobs complete." and skip the sync daemon entirely (W&B uploaded live during each run).

```python
if USE_SLURM:
    print("All jobs submitted. Running Weights & Biases sync daemon. Ctrl+C to stop syncing")
    subprocess.run(
        ["python", "wandb-sync-daemon.py", "--save_dirs", str(save_dirs_file)],
        check=True,
    )
else:
    print("All jobs complete.")
```

{{2. Confirmed: local mode skips the sync daemon and omits `--wandb_not_upload` so W&B uploads directly as each job runs.}}
---

## Files

- [x] ~~`slurm_run_blobs_deep_linear.py`~~
- [x] ~~`slurm_run_cifar_3_deep_linear.py`~~
- [x] ~~`slurm_run_mnist_deep_linear.py`~~

---

## Notes

- `USE_SLURM = True`: jobs submitted to SLURM with `--wandb_not_upload`; sync daemon runs after.
- `USE_SLURM = False`: jobs run serially in the current process without `--wandb_not_upload`; W&B uploads live; no sync daemon.
- `perform_downloads.py` calls (cifar3, mnist scripts only) are unaffected — they run before job submission regardless of mode.
- The blobs-specific teacher model check is also unaffected.
- No new files needed.
