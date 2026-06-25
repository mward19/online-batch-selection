"""Run-directory identity, collision rules, and resume plumbing.

Every run owns one self-contained directory under ``./experiments/``. The
directory is claimed atomically so parallel SLURM jobs cannot collide, and all
run outputs are written through ``write_guard`` so an accidental overwrite is
loud rather than silent. The only sanctioned overwrites are the §3 label cache
and the rolling checkpoint written via ``atomic_save``.

See plans/plan_spring_cleaning.md §1, §2, §9.
"""

import os
import shutil
from datetime import datetime

EXPERIMENTS_ROOT = "experiments"
SLURM_HISTORY_DIRNAME = "slurm_history"
RUN_SUBDIRS = ("wandb", "logs", "snapshots")
LABELS_LINK_NAME = "labels"
RESUMED_FROM_LINK_NAME = "resumed_from"


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slurm_job_id():
    return os.environ.get("SLURM_JOB_ID")


def _slurm_history_dir(experiments_root):
    return os.path.join(experiments_root, SLURM_HISTORY_DIRNAME)


def _claim_dir(parent, base_name):
    """Atomically claim ``parent/base_name``, adding a ``_<n>`` suffix on
    collision. The atomic ``os.makedirs(exist_ok=False)`` *is* the collision
    check, so there is no TOCTOU race between parallel jobs."""
    candidate = os.path.join(parent, base_name)
    n = 0
    while True:
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            n += 1
            candidate = os.path.join(parent, f"{base_name}_{n}")


def _make_subdirs(run_dir):
    for sub in RUN_SUBDIRS:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)


def _register_slurm_pointer(experiments_root, job_id, run_dir):
    history_dir = _slurm_history_dir(experiments_root)
    os.makedirs(history_dir, exist_ok=True)
    pointer = os.path.join(history_dir, job_id)
    target = os.path.relpath(run_dir, history_dir)
    try:
        os.symlink(target, pointer)
    except FileExistsError:
        pass


def _copy_run_contents(src, dst):
    """Copy a parent run's contents into a freshly claimed dir, skipping the
    ``labels`` symlink (re-created fresh per §3). This populates a brand-new
    directory from a trusted source, so it does not go through write_guard."""
    for name in os.listdir(src):
        if name in (LABELS_LINK_NAME, RESUMED_FROM_LINK_NAME):
            continue
        source = os.path.join(src, name)
        dest = os.path.join(dst, name)
        if os.path.isdir(source) and not os.path.islink(source):
            shutil.copytree(source, dest)
        else:
            shutil.copy2(source, dest, follow_symlinks=False)
    _make_subdirs(dst)


def _write_resumed_from(run_dir, parent_dir):
    link = os.path.join(run_dir, RESUMED_FROM_LINK_NAME)
    os.symlink(os.path.relpath(parent_dir, run_dir), link)


def setup_run_dir(experiments_root=EXPERIMENTS_ROOT, resume_from=None):
    """Resolve the run directory for this process.

    Returns ``(run_dir, mode, info)`` where ``mode`` is one of:

    - ``"extension"``: a new dir forked from ``resume_from`` with the parent's
      contents copied in (§9.3).
    - ``"restart"``: the existing dir for this SLURM job id, reused after a
      requeue (§9.2).
    - ``"fresh"``: a brand-new run dir.
    """
    os.makedirs(experiments_root, exist_ok=True)

    if resume_from:
        parent_dir = os.path.abspath(resume_from)
        if not os.path.isdir(parent_dir):
            raise FileNotFoundError(f"Resume parent run dir not found: {parent_dir}")
        run_dir = _claim_dir(experiments_root, f"run_{_timestamp()}")
        _copy_run_contents(parent_dir, run_dir)
        _write_resumed_from(run_dir, parent_dir)
        return run_dir, "extension", {"parent_dir": parent_dir}

    job_id = _slurm_job_id()
    if job_id is not None:
        pointer = os.path.join(_slurm_history_dir(experiments_root), job_id)
        if os.path.lexists(pointer):
            return os.path.realpath(pointer), "restart", {}

    run_dir = _claim_dir(experiments_root, f"run_{_timestamp()}")
    _make_subdirs(run_dir)
    if job_id is not None:
        _register_slurm_pointer(experiments_root, job_id, run_dir)
    return run_dir, "fresh", {}


def write_guard(path):
    """Raise if ``path`` already exists. Used for all (non-cache) run outputs to
    enforce 'collisions are bugs, and bugs are loud' (§2)."""
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing run output: {path}")


def atomic_save(save_fn, path):
    """Write the rolling target via a temp file + ``os.replace`` so it advances
    atomically and is never observed half-written (§9.2)."""
    tmp_path = f"{path}.tmp"
    save_fn(tmp_path)
    os.replace(tmp_path, path)
