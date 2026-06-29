"""Build the diagnostics for a run from the merged config's ``diagnostics:``
subtree. Returns a ``DiagnosticsRunner`` that ``SelectionMethod``
drives at run-start/after-batch/after-epoch.
"""

import os

from methods.diagnostics.base import DiagnosticsBuilder, DiagnosticsManager, Phase, TrainState
from methods.diagnostics.standard import Checkpoint, _LossErrorLeaf
from methods.diagnostics.diagnostics import EPOCH_END_DIAGNOSTICS, POST_BATCH_DIAGNOSTICS
from methods.diagnostics.schedule import LogSchedule


class DiagnosticsRunner:
    """Thin facade over the per-phase managers for the training loop."""

    def __init__(self, post_batch_manager, epoch_end_manager, checkpoint):
        self.post_batch_manager = post_batch_manager
        self.epoch_end_manager = epoch_end_manager
        self.checkpoint = checkpoint

    def run_post_batch(self, total_steps, epoch, batch_idx, total_epochs, total_batches):
        state = TrainState(total_steps=total_steps, phase=Phase.POST_BATCH, epoch=epoch,
                           batch_idx=batch_idx, total_epochs=total_epochs, total_batches=total_batches)
        self.post_batch_manager.run_diagnostics(state)

    def run_epoch_end(self, total_steps, epoch, total_epochs):
        state = TrainState(total_steps=total_steps, phase=Phase.EPOCH_END, epoch=epoch,
                           total_epochs=total_epochs)
        self.epoch_end_manager.run_diagnostics(state)

    def finalize(self):
        for manager in (self.post_batch_manager, self.epoch_end_manager):
            for diagnostic in manager.diagnostics:
                if hasattr(diagnostic, "finalize"):
                    diagnostic.finalize()

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


def create_diagnostics(method, *, project_root, **other_resources):
    """Build the per-phase diagnostics managers for ``method`` (a
    ``SelectionMethod``). ``project_root`` is the only resource not derivable
    from ``method``; all diagnostics access method state directly via
    ``self.method``.
    """
    config = method.config
    diagnostics_config = config.get("diagnostics", {}) or {}

    defaults = diagnostics_config.get("logging_defaults", {})
    requested = diagnostics_config.get("diagnostics", {}) or {}

    default_schedule = _build_schedule(method, defaults)
    logs_dir = os.path.join(config['save_dir'], "logs")

    builder = DiagnosticsBuilder()
    post_batch_manager = DiagnosticsManager(method=method, project_root=project_root)
    epoch_end_manager = DiagnosticsManager(method=method, project_root=project_root)
    checkpoint = None
    # Guard against two enabled loss/acc leaves resolving to the same W&B metric
    # name (their wandb.log calls would clobber each other at the same step).
    seen_log_keys = {}

    for name, spec in requested.items():
        spec = spec or {}
        params = dict(spec.get("params", {}))
        params.setdefault("log_path", os.path.join(logs_dir, f"{name}.log"))
        schedule = _build_schedule(method, {**defaults, **spec["logging"]}) if spec.get("logging") else default_schedule

        if name in POST_BATCH_DIAGNOSTICS:
            cls = POST_BATCH_DIAGNOSTICS[name]
            diagnostic = cls(post_batch_manager, builder, should_run=schedule, **params)
            post_batch_manager.register(diagnostic)
            if cls is Checkpoint:
                checkpoint = diagnostic
        elif name in EPOCH_END_DIAGNOSTICS:
            cls = EPOCH_END_DIAGNOSTICS[name]
            diagnostic = cls(epoch_end_manager, builder, should_run=(lambda state: True), **params)
            epoch_end_manager.register(diagnostic)
        else:
            raise ValueError(f"Unknown diagnostic '{name}' in config diagnostics.diagnostics.")

        if isinstance(diagnostic, _LossErrorLeaf):
            clash = seen_log_keys.get(diagnostic.log_key)
            if clash is not None:
                raise ValueError(
                    f"Diagnostics '{clash}' and '{name}' both resolve to log_key "
                    f"'{diagnostic.log_key}'; enabled loss/acc leaves must have distinct "
                    "log_keys (set a 'log_key' param to disambiguate)."
                )
            seen_log_keys[diagnostic.log_key] = name

    return DiagnosticsRunner(post_batch_manager, epoch_end_manager, checkpoint)
