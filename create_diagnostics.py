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
    _LossErrorLeaf,
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


def create_diagnostics(method, *, project_root, **other_resources):
    """Build the per-phase diagnostics managers for ``method`` (a
    ``SelectionMethod``). All static run resources are extracted from ``method``
    here; ``project_root`` is the only one not derivable from the config.
    ``other_resources`` lets subclasses inject extra static context. The
    resources seed each manager's ``static_context`` and are used here to build
    the schedules and log paths. Per-step values (model, device, …) arrive later
    via shared_context.
    """
    config = method.config
    diagnostics_config = config.get("diagnostics", {}) or {}

    resources = {
        'save_dir':           config['save_dir'],
        'project_root':       project_root,
        'dataset_name':       config['dataset']['name'],
        'model_name':         config['networks']['params'].get('m_type', config['networks']['type']),
        'seed':               config['seed'],
        'fixed_train_loader': method.fixed_train_loader,
        'test_loader':        method.test_loader,
        'total_batches':      len(method.train_loader),
        'num_train_samples':  method.num_train_samples,
        'num_epochs':         method.epochs,
        'num_steps':          method.num_steps,
        'initial_best_acc':   method.best_acc,
        'initial_best_epoch': method.best_epoch,
        'noisy_indices':      method.data_info.get('noisy_indices'),
        'true_labels':        method.data_info.get('true_labels'),
        'wstar_test_acc':     method.data_info.get('wstar_test_acc'),
        'what_test_acc':      method.data_info.get('what_test_acc'),
        'bayes_accuracy':     config.get('bayes_accuracy'),
        'num_classes':        method.num_classes,
        'config':             config,
        'logger':             method.logger,
        **other_resources,
    }

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
    # Guard against two enabled loss/acc leaves resolving to the same W&B metric
    # name (their wandb.log calls would clobber each other at the same step).
    seen_log_keys = {}

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
