"""Build the diagnostics for a run from the merged config's ``diagnostics:``
subtree. Returns a ``DiagnosticsRunner`` that ``SelectionMethod``
drives at run-start/after-batch/after-epoch.
"""

import os

from methods.diagnostics.base import DiagnosticsManager, Phase, TrainState
from methods.diagnostics.standard import Checkpoint, _LossErrorLeaf
from methods.diagnostics.diagnostics import EPOCH_END_DIAGNOSTICS, POST_BATCH_DIAGNOSTICS, TRAIN_END_DIAGNOSTICS
from methods.diagnostics.schedule import LogSchedule


class DiagnosticsRunner:
    """Thin facade over the per-phase managers for the training loop."""

    def __init__(self, post_batch_manager, epoch_end_manager, train_end_manager, checkpoint):
        self.post_batch_manager = post_batch_manager
        self.epoch_end_manager = epoch_end_manager
        self.train_end_manager = train_end_manager
        self.checkpoint = checkpoint
        self._last_total_steps = 0

    def run_post_batch(self, total_steps, epoch, batch_idx, total_epochs, total_batches):
        self._last_total_steps = total_steps
        state = TrainState(total_steps=total_steps, phase=Phase.POST_BATCH, epoch=epoch,
                           batch_idx=batch_idx, total_epochs=total_epochs, total_batches=total_batches)
        self.post_batch_manager.run_diagnostics(state)

    def run_epoch_end(self, total_steps, epoch, total_epochs):
        self._last_total_steps = total_steps
        state = TrainState(total_steps=total_steps, phase=Phase.EPOCH_END, epoch=epoch,
                           total_epochs=total_epochs)
        self.epoch_end_manager.run_diagnostics(state)

    def finalize(self):
        for manager in (self.post_batch_manager, self.epoch_end_manager):
            for diagnostic in manager.diagnostics:
                if hasattr(diagnostic, "finalize"):
                    diagnostic.finalize()
        state = TrainState(total_steps=self._last_total_steps, phase=Phase.TRAIN_END)
        self.train_end_manager.run_diagnostics(state)

    @property
    def best_acc(self):
        return self.checkpoint.best_acc if self.checkpoint is not None else 0.0

    @property
    def best_epoch(self):
        return self.checkpoint.best_epoch if self.checkpoint is not None else 0

    @property
    def is_best(self):
        return self.checkpoint.is_best if self.checkpoint is not None else False


def _build_schedule(method, logging):
    return LogSchedule(
        total_batches=len(method.train_loader),
        num_epochs=method.epochs,
        num_steps=method.num_steps,
        log_interval=logging.get("log_interval", "logarithmic"),
        save_init=logging.get("save_init", 5),
        save_freq=logging.get("save_freq", 4),
    )


def _parse_diagnostics_list(requested):
    """Parse a list of {ClassName: params_dict} entries into (cls_name, log_name, params) tuples.

    When the same class appears more than once, log_name gets a _0, _1, ... suffix.
    """
    if not isinstance(requested, list):
        raise ValueError(
            "diagnostics.diagnostics must be a list of ClassName or {ClassName: params} entries."
        )
    type_counts = {}
    for entry in requested:
        # bare string ("- TrainLoss") for no-param diagnostics; dict for params
        if isinstance(entry, str):
            cls_name = entry
        elif isinstance(entry, dict) and len(entry) == 1:
            cls_name = next(iter(entry))
        else:
            raise ValueError(
                f"Each diagnostics entry must be a class name string or single-key dict, got: {entry!r}"
            )
        type_counts[cls_name] = type_counts.get(cls_name, 0) + 1

    seen = {}
    result = []
    for entry in requested:
        if isinstance(entry, str):
            cls_name, params = entry, {}
        else:
            cls_name = next(iter(entry))
            params = dict(entry[cls_name] or {})
        if type_counts[cls_name] > 1:
            n = seen.get(cls_name, 0)
            log_name = f"{cls_name}_{n}"
            seen[cls_name] = n + 1
        else:
            log_name = cls_name
        result.append((cls_name, log_name, params))
    return result


def create_diagnostics(method, *, project_root, **other_resources):
    """Build the per-phase diagnostics managers for ``method`` (a
    ``SelectionMethod``). ``project_root`` is the only resource not derivable
    from ``method``; all diagnostics access method state directly via
    ``self.method``.
    """
    config = method.config
    diagnostics_config = config.get("diagnostics", {}) or {}

    defaults = diagnostics_config.get("logging_defaults", {})
    requested = diagnostics_config.get("diagnostics", []) or []

    default_schedule = _build_schedule(method, defaults)
    logs_dir = os.path.join(config['save_dir'], "logs")

    post_batch_manager = DiagnosticsManager(method=method, project_root=project_root)
    epoch_end_manager = DiagnosticsManager(method=method, project_root=project_root)
    train_end_manager = DiagnosticsManager(method=method, project_root=project_root)
    checkpoint = None
    # Guard against two enabled loss/acc leaves resolving to the same W&B metric
    # name (their wandb.log calls would clobber each other at the same step).
    seen_log_keys = {}

    for cls_name, log_name, params in _parse_diagnostics_list(requested):
        params.setdefault("log_path", os.path.join(logs_dir, f"{log_name}.log"))

        if cls_name in POST_BATCH_DIAGNOSTICS:
            cls = POST_BATCH_DIAGNOSTICS[cls_name]
            diagnostic = cls(post_batch_manager, should_run=default_schedule, **params)
            post_batch_manager.register(diagnostic)
            if cls is Checkpoint:
                checkpoint = diagnostic
        elif cls_name in EPOCH_END_DIAGNOSTICS:
            cls = EPOCH_END_DIAGNOSTICS[cls_name]
            diagnostic = cls(epoch_end_manager, should_run=(lambda state: True), **params)
            epoch_end_manager.register(diagnostic)
        elif cls_name in TRAIN_END_DIAGNOSTICS:
            cls = TRAIN_END_DIAGNOSTICS[cls_name]
            diagnostic = cls(train_end_manager, should_run=(lambda state: True), **params)
            train_end_manager.register(diagnostic)
        else:
            raise ValueError(f"Unknown diagnostic '{cls_name}' in config diagnostics.diagnostics.")

        if isinstance(diagnostic, _LossErrorLeaf):
            clash = seen_log_keys.get(diagnostic.log_key)
            if clash is not None:
                raise ValueError(
                    f"Diagnostics '{clash}' and '{log_name}' both resolve to log_key "
                    f"'{diagnostic.log_key}'; enabled loss/acc leaves must have distinct "
                    "log_keys (set a 'log_key' param to disambiguate)."
                )
            seen_log_keys[diagnostic.log_key] = log_name

    return DiagnosticsRunner(post_batch_manager, epoch_end_manager, train_end_manager, checkpoint)
