# Plan: link SLURM stdout/stderr into the run dir

~~Implemented.~~

## Goal

Make each run's SLURM logs reachable from its own directory. When a run executes
under SLURM, create two symlinks inside `<run_dir>/logs/`:

```
slurm_<job_id>.out -> ../../../logs/slurm/<job_id>.out
slurm_<job_id>.err -> ../../../logs/slurm/<job_id>.err
```

(Verified by hand against
`experiments/20260626_122711_run_MNIST_Noise_Uniform_SGD_lr-0.1_CrossEntropy`;
both resolve correctly.)

## Why a symlink (not a move/copy)

- The SLURM run dir name carries a runtime timestamp + collision suffix
  (`_claim_dir`), so `#SBATCH --output` cannot point at it directly — the path is
  unknown at submission time.
- SLURM keeps the log file open and writing for the whole job (plus epilog). A
  symlink leaves the real file in `logs/slurm/` and exposes it live from the run
  dir, capturing the complete log with no race and no fd issues.
- Including `<job_id>` in the link name means a requeue/restart (new job id,
  same run dir) adds a second pair of links rather than clobbering the first —
  one link per attempt.

## Constraints / assumptions

- The submit scripts write SLURM logs to `logs/slurm/%j.{out,err}` relative to
  the submission cwd (repo root), e.g. `run_mnist.py:49-50`. `main.py` also runs
  from the repo root, and `experiments_root` is the relative `"experiments"`
  default. So from `<run_dir>/logs/`, the repo-root `logs/slurm/` dir is three
  hops up. The implementation computes this with `os.path.relpath` rather than
  hardcoding `../../../`, so it stays correct if `experiments_root` ever changes
  depth.

## Changes (all in `run_dir.py`)

### 1. Add a constant for the SLURM log dir

Next to the other constants near the top (`run_dir.py:16-20`):

```python
# Mirrors the logs/slurm/%j path the submit scripts pass to #SBATCH
# --output/--error (e.g. run_mnist.py). Kept in sync by hand.
SLURM_LOG_DIR = os.path.join("logs", "slurm")
```

### 2. Add a helper to create the links

Place near `_register_slurm_pointer` (`run_dir.py:114-122`), following its
`try/except FileExistsError` idiom so it is idempotent:

```python
def _link_slurm_logs(run_dir, job_id):
    """Symlink this job's SLURM stdout/stderr into ``<run_dir>/logs`` so the
    logs are reachable from the run dir. The real files stay in
    ``logs/slurm/`` (SLURM keeps writing to them); these are just live links.
    Named ``slurm_<job_id>.{out,err}`` so a requeue (new job id, same run dir)
    adds a fresh pair instead of clobbering the previous attempt."""
    logs_dir = os.path.join(run_dir, "logs")
    for ext in ("out", "err"):
        real = os.path.join(SLURM_LOG_DIR, f"{job_id}.{ext}")
        link = os.path.join(logs_dir, f"slurm_{job_id}.{ext}")
        target = os.path.relpath(real, logs_dir)
        try:
            os.symlink(target, link)
        except FileExistsError:
            pass
```

Note: the link is created whether or not the real file exists yet — a dangling
symlink is fine and resolves as soon as SLURM creates the file. `logs_dir`
already exists because `_make_subdirs` ran (fresh mode) or the dir was populated
on a prior attempt (restart mode).

### 3. Call it from `setup_run_dir`

In the `job_id is not None` branches of `setup_run_dir`
(`run_dir.py:170-180`):

- **fresh** mode: after `_register_slurm_pointer(...)` at `run_dir.py:179`, call
  `_link_slurm_logs(run_dir, job_id)`.
- **restart** mode (`run_dir.py:173-174`): before returning the reused dir, call
  `_link_slurm_logs(os.path.realpath(pointer), job_id)`. With `--requeue` the job
  id is preserved, so this is usually a harmless no-op (link exists); it is kept
  for safety/idempotence.

`extension` mode (manual `--resume_from`, `run_dir.py:161-168`) does not run
under a fresh SLURM allocation in the normal flow; leave it unchanged. {{If you
want extension runs linked too, say so and I'll add the call after
`_copy_run_contents`.}}

## Out of scope

- No change to the submit scripts (`run_mnist.py` etc.) — they keep writing to
  `logs/slurm/%j.{out,err}`.
- No cleanup of the originals in `logs/slurm/`.

## Manual verification after implementing

1. Remove the hand-made example links:
   `rm experiments/20260626_122711_run_MNIST_Noise_Uniform_SGD_lr-0.1_CrossEntropy/logs/slurm_12424049.{out,err}`
2. Submit a small sweep (e.g. `run_mnist.py`) and confirm each run dir's `logs/`
   gets `slurm_<job_id>.{out,err}` resolving (`readlink -e`) to the files in
   `logs/slurm/`.


## other things to do
Rename configs-temp to .temp-configs so it's hidden