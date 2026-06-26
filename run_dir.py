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


CONFIG_REF_SIGIL = "$"


def _resolve_config_ref(config, path):
    """Resolve a dotted config path; raise loudly if any segment is absent."""
    cur = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(
                f"run_name_format references config path '{path}', but segment "
                f"'{part}' is not present in the config."
            )
        cur = cur[part]
    return cur


def _render_run_name_token(config, token):
    """A ``$``-prefixed string is a config reference (resolved, raises on a bad
    path); any other scalar is a literal."""
    if isinstance(token, str) and token.startswith(CONFIG_REF_SIGIL):
        return str(_resolve_config_ref(config, token[len(CONFIG_REF_SIGIL):]))
    return str(token)


def build_run_name(config, fmt, *, value_sep="_", kv_sep="-"):
    """Render ``run_name_format`` (a required list) into a run name string.

    Each element is a literal string, a ``$dotted.path`` config reference, or a
    single-key dict ``{label: [tokens...]}`` rendered as
    ``label<kv_sep>value_sep.join(values)`` (e.g. ``lr-0.001``). ``value_sep``
    joins top-level elements and the values within a key-value pair; ``kv_sep``
    joins a pair's label to its value block. Both are parameters so the scheme
    can change later.
    """
    if fmt is None:
        raise ValueError(
            "run_name_format is required in the config but was not provided."
        )
    if not isinstance(fmt, list):
        raise ValueError(f"run_name_format must be a list, got {type(fmt).__name__}.")

    parts = []
    for element in fmt:
        if isinstance(element, dict):
            if len(element) != 1:
                raise ValueError(
                    f"run_name_format key-value element must have exactly one key: {element!r}"
                )
            (label, tokens), = element.items()
            tokens = tokens if isinstance(tokens, list) else [tokens]
            values = [_render_run_name_token(config, t) for t in tokens]
            parts.append(f"{label}{kv_sep}{value_sep.join(values)}")
        else:
            parts.append(_render_run_name_token(config, element))
    return value_sep.join(parts)


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slurm_job_id():
    return os.environ.get("SLURM_JOB_ID")


def _slurm_history_dir(experiments_root):
    return os.path.join(experiments_root, SLURM_HISTORY_DIRNAME)


def _claim_dir(parent, timestamp, run_name):
    """Atomically claim ``parent/<timestamp>_<run_name>``, inserting a ``_<n>``
    suffix **right after the timestamp** on collision (e.g.
    ``20260626_110000_1_run_CIFAR3_...``). The atomic
    ``os.makedirs(exist_ok=False)`` *is* the collision check, so there is no
    TOCTOU race between parallel jobs."""
    n = 0
    while True:
        stamp = timestamp if n == 0 else f"{timestamp}_{n}"
        candidate = os.path.join(parent, f"{stamp}_{run_name}")
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            n += 1


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


def setup_run_dir(run_name, experiments_root=EXPERIMENTS_ROOT, resume_from=None):
    """Resolve the run directory for this process.

    The directory is named ``<timestamp>[_<n>]_<run_name>`` (collision suffix
    right after the timestamp). Returns ``(run_dir, mode, info)`` where ``mode``
    is one of:

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
        run_dir = _claim_dir(experiments_root, _timestamp(), run_name)
        _copy_run_contents(parent_dir, run_dir)
        _write_resumed_from(run_dir, parent_dir)
        return run_dir, "extension", {"parent_dir": parent_dir}

    job_id = _slurm_job_id()
    if job_id is not None:
        pointer = os.path.join(_slurm_history_dir(experiments_root), job_id)
        if os.path.lexists(pointer):
            return os.path.realpath(pointer), "restart", {}

    run_dir = _claim_dir(experiments_root, _timestamp(), run_name)
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
