import torch
import wandb
import numpy as np

from methods.method_utils.diagnostics_context import DiagnosticsRunContext
from methods.method_utils.ntk import NTKDiagnostics
from methods.method_utils.param_grad import ParamGradDiagnostics
from methods.method_utils.probe import ProbeDiagnostics
from methods.method_utils.snapshots import SnapshotManager
from methods.method_utils.weight_matrix import WeightMatrixDiagnostics

from datetime import timedelta


class DiagnosticsLogger:
    def __init__(
        self,
        logger,
        num_classes,
        criterion,
        diagnostics_config,
        run_config,
        context: DiagnosticsRunContext,
        model=None,
    ):
        self.logger = logger
        self.num_classes = num_classes
        self.criterion = criterion

        diagnostics_config = diagnostics_config or {}
        self.total_batches = int(context.total_batches)
        self.num_epochs = context.num_epochs
        self.num_steps = context.num_steps
        self.log_interval = diagnostics_config.get('log_interval', 'logarithmic')
        self.save_init = int(diagnostics_config.get('save_init', 5))
        self.save_freq = int(diagnostics_config.get('save_freq', 4))
        if self.log_interval not in {'logarithmic', 'per_epoch'}:
            self.logger.info(
                f"Warning: unknown diagnostics log_interval '{self.log_interval}'. Falling back to 'logarithmic'."
            )
            self.log_interval = 'logarithmic'
        self.last_batch_idx = self.total_batches - 1
        self._logged_steps = set()
        self._logarithmic_steps = self._build_logarithmic_steps()

        self.local_snapshots = bool(diagnostics_config.get('local_snapshots', False))
        self.local_points_selected = bool(diagnostics_config.get('local_points_selected', False))
        self.local_best_model_checkpoints = bool(diagnostics_config.get('local_best_model_checkpoints', False))
        self.wandb_loss_acc = bool(diagnostics_config.get('wandb_loss_acc', False))
        self.wandb_wstar_acc = bool(diagnostics_config.get('wandb_wstar_acc', False))
        self.wstar_test_acc  = context.wstar_test_acc
        self.what_test_acc   = context.what_test_acc
        self.bayes_accuracy  = context.bayes_accuracy
        self.wandb_normed_logits = bool(diagnostics_config.get('wandb_normed_logits', False))
        self.wandb_param_norms = bool(diagnostics_config.get('wandb_param_norms', False))
        self.wandb_weight_matrix_norms = bool(diagnostics_config.get('wandb_weight_matrix_norms', False))
        self.wandb_grad_norms = bool(diagnostics_config.get('wandb_grad_norms', False))
        self.wandb_linear_probe = bool(diagnostics_config.get('wandb_linear_probe', False))
        self.wandb_ntk = bool(diagnostics_config.get('wandb_ntk', False))
        self.lr_max_iter = int(diagnostics_config.get('linear_probe_max_iter', 300))
        self.lr_max_samples = int(diagnostics_config.get('linear_probe_max_samples', -1))
        ntk_max_samples = int(diagnostics_config.get('ntk_max_samples', 1000))
        ntk_top_k = int(diagnostics_config.get('ntk_top_k', 10))
        ntk_variant = str(diagnostics_config.get('ntk_variant', diagnostics_config.get('ntk_kernel_type', 'pntk')))
        ntk_eigenvalue_concentration_checkpoints = diagnostics_config.get(
            'ntk_eigenvalue_concentration_checkpoints',
            [20, 40, 80],
        )
        local_spectrum = bool(diagnostics_config.get('local_spectrum', False))
        self.param_grad_diagnostics = ParamGradDiagnostics(
            wandb_param_norms=self.wandb_param_norms,
            wandb_grad_norms=self.wandb_grad_norms,
            logger=self.logger,
        )
        self.should_log_param_stats = self.param_grad_diagnostics.enabled
        self.should_build_snapshot = (
            self.local_snapshots
            or self.wandb_loss_acc
            or self.wandb_normed_logits
            or self.local_best_model_checkpoints
        )
        self.snapshot_manager = SnapshotManager(
            logger=self.logger,
            context=context,
            should_track_points=self.local_points_selected,
            should_save_snapshots=self.local_snapshots,
            should_save_best_checkpoint=self.local_best_model_checkpoints,
            initial_best_acc=context.initial_best_acc,
            initial_best_epoch=context.initial_best_epoch,
            checkpoint_saver=context.checkpoint_saver,
        )
        self.probe_diagnostics = ProbeDiagnostics(
            logger=self.logger,
            context=context,
            lr_max_iter=self.lr_max_iter,
            lr_max_samples=self.lr_max_samples,
            enabled=self.wandb_linear_probe,
        )
        self.ntk_diagnostics = NTKDiagnostics(
            logger=self.logger,
            context=context,
            num_classes=self.num_classes,
            ntk_max_samples=ntk_max_samples,
            ntk_top_k=ntk_top_k,
            ntk_variant=ntk_variant,
            ntk_eigenvalue_concentration_checkpoints=ntk_eigenvalue_concentration_checkpoints,
            enabled=self.wandb_ntk,
            config=run_config,
            save_spectrum=local_spectrum,
        )
        weight_matrix_param_names = diagnostics_config.get('weight_matrix_param_names') or None
        weight_matrix_last_n_layers = diagnostics_config.get('weight_matrix_last_n_layers')
        weight_matrix_last_n_layers = int(weight_matrix_last_n_layers) if weight_matrix_last_n_layers is not None else None
        self.weight_matrix_diagnostics = WeightMatrixDiagnostics(
            logger=self.logger,
            context=context,
            enabled=self.wandb_weight_matrix_norms,
            param_names=weight_matrix_param_names,
            last_n_layers=weight_matrix_last_n_layers,
        )
        self.should_log_probe = self.probe_diagnostics.enabled
        self.should_log_ntk = self.ntk_diagnostics.enabled
        self.snapshots = self.snapshot_manager.snapshots

        if self.wandb_ntk and not self.should_log_ntk:
            self.logger.info('Warning: disabling NTK diagnostics because ntk_max_samples or ntk_top_k is non-positive.')

        if self.wandb_wstar_acc and self.wstar_test_acc is not None:
            wandb.summary['wstar_test_acc'] = self.wstar_test_acc
        if self.what_test_acc is not None:
            wandb.summary['what_test_acc'] = self.what_test_acc
        if self.bayes_accuracy is not None:
            wandb.summary['bayes_accuracy'] = self.bayes_accuracy

    @property
    def best_acc(self):
        return self.snapshot_manager.best_acc

    @property
    def best_epoch(self):
        return self.snapshot_manager.best_epoch

    def _build_logarithmic_steps(self):
        """Creates set of checkpoint steps to log at if logging on logarithmic scale"""
        total_epochs = self.num_epochs or int(np.ceil(self.num_steps / self.total_batches))
        intra_epoch_stride = max(self.total_batches // self.save_freq, 1)

        t = 0
        steps = [0]
        for epoch in range(total_epochs):
            for batch_idx in range(self.total_batches):
                t += 1
                if epoch < self.save_init and batch_idx % intra_epoch_stride == 0:
                    steps.append(t)

            if self.save_init <= epoch <= 25:
                steps.append(t)
            elif 25 < epoch <= 65 and epoch % 4 == 0:
                steps.append(t)
            elif (epoch > 65 and epoch % 15 == 0) or (epoch == total_epochs - 1):
                steps.append(t)

        if self.num_steps is not None:
            steps = [step for step in steps if step <= self.num_steps]

        return set(steps)

    def should_log(self, total_step, batch_idx, force=False):
        """Returns True if current step is a checkpoint we log at"""
        if force:
            return True
        if total_step in self._logged_steps:
            return False
        if self.log_interval == 'per_epoch':
            return batch_idx == self.last_batch_idx
        return total_step in self._logarithmic_steps

    def mark_logged(self, total_step):
        """In case we call log_diagnostics multiple times on the same step in SelectionMethod.py 
        (i.e. call after each batch and also after each epoch), then we avoid duplicates"""
        self._logged_steps.add(total_step)

    def log_diagnostics(
        self,
        model,
        trigger,
        total_step,
        epoch,
        batch_idx,
        device,
        selected_indexes=None,
        lr=None,
        total_time=None,
        time_this_epoch=None,
        checkpoint_state=None,
        force=False,
    ):
        """Logs all snapshots, linear probing, parameter and gradient norms, and NTK metrics specified in diagnostics config file"""
        """Called once at initialization and then after every batch; only logs at specified logging checkpoints (logarithmic step or epoch scale)"""
        """Note that Snapshots is much more complicated than the other calls"""
        was_training = model.training
        self.snapshot_manager.record_selected_points(epoch, selected_indexes)

        if not self.should_log(total_step=total_step, batch_idx=batch_idx, force=force):
            return {
                'logged': False,
                'is_best': False,
                'best_acc': self.best_acc,
                'best_epoch': self.best_epoch,
            }

        self.mark_logged(total_step)
        model.eval()

        log_data = {
            'diagnostics/trigger': trigger,
            'diagnostics/total_step': total_step,
        }
        log_data['diagnostics/epoch'] = int(epoch)

        snapshot_metrics = None
        if self.should_build_snapshot: # If we are logging snapshots and we should log on this step
            snapshot, snapshot_metrics = self.snapshot_manager.build_snapshot(model, device, total_step, epoch)
            self.snapshot_manager.store_snapshot(snapshot, total_step)
            is_best = self.snapshot_manager.update_best_checkpoint(epoch, snapshot_metrics, checkpoint_state)
            self.snapshot_manager.log_summary(epoch, total_step, lr, total_time, time_this_epoch, snapshot_metrics)

            noisy_train_metrics = None
            if self.snapshot_manager.has_label_noise() and 'train_loss_noisy_labels' in snapshot_metrics:
                noisy_train_metrics = {
                    'train_loss_noisy_labels': snapshot_metrics['train_loss_noisy_labels'],
                    'train_acc_noisy_labels': snapshot_metrics['train_acc_noisy_labels'],
                }
                self.logger.info(
                    f'=====> Epoch: {epoch}, '
                    f'train_loss_noisy_labels: {noisy_train_metrics["train_loss_noisy_labels"]:.4f}, '
                    f'train_acc_noisy_labels: {noisy_train_metrics["train_acc_noisy_labels"]:.4f}'
                )

            if self.wandb_loss_acc:
                log_data['train_loss'] = snapshot_metrics['train_loss']
                log_data['train_acc'] = snapshot_metrics['train_acc']
<<<<<<< HEAD
                # log_data['train_loss_train_loader_labels'] = snapshot_metrics['train_loss']
                if self.snapshot_manager.uses_true_labels_for_train_accuracy():
                    log_data['train_acc_true_labels'] = snapshot_metrics['train_acc_true_labels']
                    log_data['train_loss_true_labels'] = snapshot_metrics['train_loss_true_labels']
                    log_data['train_acc_loader_labels'] = snapshot_metrics['train_acc']
                    log_data['train_loss_loader_labels'] = snapshot_metrics['train_loss']
=======
                if noisy_train_metrics is not None:
                    log_data['train_loss_noisy_labels'] = noisy_train_metrics['train_loss_noisy_labels']
                    log_data['train_acc_noisy_labels'] = noisy_train_metrics['train_acc_noisy_labels']
>>>>>>> main
                log_data['val_loss'] = snapshot_metrics['val_loss']
                log_data['val_acc'] = snapshot_metrics['val_acc']
                log_data['best_val_acc'] = float(self.best_acc)
                log_data['epoch'] = int(epoch)
                log_data['lr'] = float(lr)
                log_data['total_time'] = float(total_time)
                log_data["total_time_str"] = str(timedelta(seconds=int(total_time)))
                log_data['time_epoch'] = float(time_this_epoch)

            if self.wandb_normed_logits:
                log_data['diagnostics/train_logits_norm_l2_mean'] = snapshot_metrics['train_normed_logits_l2_mean']
                log_data['diagnostics/test_logits_norm_l2_mean'] = snapshot_metrics['val_normed_logits_l2_mean']
        else:
            is_best = False


        if self.should_log_probe:
            log_data.update(self.probe_diagnostics.log_metrics(model, device))

        if self.should_log_ntk:
            log_data.update(self.ntk_diagnostics.log_metrics(model, device, total_step=total_step))
        
        if self.weight_matrix_diagnostics.enabled:
            log_data.update(self.weight_matrix_diagnostics.log_metrics(model))

        if self.should_log_param_stats:
            log_data.update(self.param_grad_diagnostics.log_metrics(model))

        self.logger.wandb_log(log_data, step=int(total_step))

        if was_training:
            model.train()

        return {
            'logged': True,
            'is_best': is_best,
            'best_acc': self.best_acc,
            'best_epoch': self.best_epoch,
        }

    def log_epoch_end_selection_stats(self, epoch, total_step):
        """Logs metrics to wanbd that must be logged at the epoch level (currently just noisy points selected)"""
        noisy_selection_stats = self.snapshot_manager.get_noisy_selection_stats()
        if noisy_selection_stats is None:
            return

        self.logger.info(
            f'=====> Epoch {epoch}: noisy selected count={noisy_selection_stats["num_noisy_selected"]}/'
            f'{noisy_selection_stats["total_noisy"]}, '
            f'fraction_of_train={noisy_selection_stats["fraction_of_train"]:.4f}, '
            f'fraction_of_noisy_pool={noisy_selection_stats["fraction_of_noisy_pool"]:.4f}'
        )
        self.logger.wandb_log(
            {
                'diagnostics/trigger': 'epoch_end_selection',
                'diagnostics/epoch': int(epoch),
                'diagnostics/total_step': int(total_step),
                'num noisy points selected': noisy_selection_stats['num_noisy_selected'],
                'percent of batch with label noise': noisy_selection_stats['fraction_of_train'],
                'percent of points with label noise selected': noisy_selection_stats['fraction_of_noisy_pool'],
            },
            step=int(total_step),
        )

    def finalize(self):
        self.ntk_diagnostics.finalize()
        self.snapshot_manager.finalize()
