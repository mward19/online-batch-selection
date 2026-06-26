"""Build the diagnostics for a run from the merged config's ``diagnostics:``
subtree (§5.6, §5.7). Returns a ``DiagnosticsRunner`` that ``SelectionMethod``
drives at run-start/after-batch/after-epoch.
"""

import os

from methods.diagnostics.base import DiagnosticsBuilder, DiagnosticsManager, Phase, TrainState
from methods.diagnostics.diagnostics import (
    Checkpoint,
    EPOCH_END_DIAGNOSTICS,
    POST_BATCH_DIAGNOSTICS,
)
from methods.diagnostics.schedule import LogSchedule


class DiagnosticsRunner:
    """Thin facade over the per-phase managers for the training loop."""

    def __init__(self, post_batch_manager, epoch_end_manager, checkpoint):
        self.post_batch_manager = post_batch_manager
        self.epoch_end_manager = epoch_end_manager
        self.checkpoint = checkpoint

    def run_post_batch(self, total_steps, epoch, batch_idx, total_epochs, total_batches, **context):
        state = TrainState(total_steps=total_steps, phase=Phase.POST_BATCH, epoch=epoch,
                           batch_idx=batch_idx, total_epochs=total_epochs, total_batches=total_batches)
        self.post_batch_manager.run_diagnostics(state, **context)

    def run_epoch_end(self, total_steps, epoch, total_epochs, **context):
        state = TrainState(total_steps=total_steps, phase=Phase.EPOCH_END, epoch=epoch,
                           total_epochs=total_epochs)
        self.epoch_end_manager.run_diagnostics(state, **context)

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


def _build_schedule(resources, logging):
    return LogSchedule(
        total_batches=resources["total_batches"],
        num_epochs=resources["num_epochs"],
        num_steps=resources["num_steps"],
        log_interval=logging.get("log_interval", "logarithmic"),
        save_init=logging.get("save_init", 5),
        save_freq=logging.get("save_freq", 4),
    )


def create_diagnostics(diagnostics_config, resources):
    """Build the per-phase diagnostics managers from the config's ``diagnostics:``
    subtree. ``resources`` is a plain dict of static run resources (loaders,
    save_dir, num_classes, true_labels, config, logger, …); it seeds each
    manager's ``static_context`` and is used here to build the schedules and log
    paths. Per-step values (model, device, …) arrive later via shared_context.
    """
    diagnostics_config = diagnostics_config or {}
    defaults = diagnostics_config.get("logging_defaults", {})
    requested = diagnostics_config.get("diagnostics", {}) or {}

    default_schedule = _build_schedule(resources, defaults)
    logs_dir = os.path.join(resources["save_dir"], "logs")

    builder = DiagnosticsBuilder()
    post_batch_manager = DiagnosticsManager()
    epoch_end_manager = DiagnosticsManager()
    post_batch_manager.set_static_context(**resources)
    epoch_end_manager.set_static_context(**resources)
    checkpoint = None

    for name, spec in requested.items():
        spec = spec or {}
        params = dict(spec.get("params", {}))
        params.setdefault("log_path", os.path.join(logs_dir, f"{name}.log"))
        schedule = _build_schedule(resources, {**defaults, **spec["logging"]}) if spec.get("logging") else default_schedule

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

    return DiagnosticsRunner(post_batch_manager, epoch_end_manager, checkpoint)
