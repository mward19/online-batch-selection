import torch
import wandb

from methods.method_utils.diagnostics_context import DiagnosticsRunContext
from methods.method_utils.ntk import NTKDiagnostics
from methods.method_utils.param_grad import ParamGradDiagnostics
from methods.method_utils.probe import ProbeDiagnostics
from methods.method_utils.snapshots import SnapshotManager
from methods.method_utils.weights import WeightDiagnostics

from datetime import timedelta


class DiagnosticsLogger:
    def __init__(
        self,
        logger,
        num_classes,
        criterion,
        diagnostics_config,
        context: DiagnosticsRunContext,
    ):
        self.logger = logger
        self.num_classes = num_classes
        self.criterion = criterion

        diagnostics_config = diagnostics_config or {}
        self.local_snapshots = bool(diagnostics_config.get('local_snapshots', False))
        self.local_points_selected = bool(diagnostics_config.get('local_points_selected', False))
        self.local_best_model_checkpoints = bool(diagnostics_config.get('local_best_model_checkpoints', False))
        self.wandb_loss_acc = bool(diagnostics_config.get('wandb_loss_acc', False))
        self.wandb_normed_logits = bool(diagnostics_config.get('wandb_normed_logits', False))
        self.wandb_param_norms = bool(diagnostics_config.get('wandb_param_norms', False))
        self.wandb_grad_norms = bool(diagnostics_config.get('wandb_grad_norms', False))
        self.wandb_linear_probe = bool(diagnostics_config.get('wandb_linear_probe', False))
        self.wandb_ntk = bool(diagnostics_config.get('wandb_ntk', False))
        self.histogram_max_points = int(diagnostics_config.get('wandb_histogram_max_points', 200000))
        self.lr_max_iter = int(diagnostics_config.get('linear_probe_max_iter', 300))
        self.layer_names = list(diagnostics_config.get('layers', []))
        ntk_max_samples = diagnostics_config.get('ntk_max_samples')
        ntk_max_samples = 1000 if ntk_max_samples is None else int(ntk_max_samples)
        ntk_top_k = int(diagnostics_config.get('ntk_top_k', 10))
        local_spectral_decay = bool(diagnostics_config.get('local_spectral_decay', False))
        teacher_model_config = diagnostics_config.get('teacher_model_config')
        self.param_grad_diagnostics = ParamGradDiagnostics(
            wandb_param_norms=self.wandb_param_norms,
            wandb_grad_norms=self.wandb_grad_norms,
            histogram_max_points=self.histogram_max_points,
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
            log_interval=diagnostics_config.get('log_interval', 'logarithmic'),
            save_init=diagnostics_config.get('save_init', 5),
            save_freq=diagnostics_config.get('save_freq', 4),
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
            layer_names=self.layer_names,
            lr_max_iter=self.lr_max_iter,
        )
        self.ntk_diagnostics = NTKDiagnostics(
            logger=self.logger,
            context=context,
            num_classes=self.num_classes,
            ntk_max_samples=ntk_max_samples,
            ntk_top_k=ntk_top_k,
            enabled=self.wandb_ntk and ntk_max_samples > 0 and ntk_top_k > 0,
            teacher_model_config=teacher_model_config,
            save_spectral_decay=local_spectral_decay,
        )
        self.weight_matrix_diagnostics = WeightDiagnostics(
            logger=self.logger,
            context=context,
            enabled=True, # TEMP
        )
        self.should_log_probe = self.probe_diagnostics.enabled
        self.should_log_ntk = self.ntk_diagnostics.enabled
        self.snapshots = self.snapshot_manager.snapshots

        if self.wandb_ntk and not self.should_log_ntk:
            self.logger.info('Warning: disabling NTK diagnostics because ntk_max_samples or ntk_top_k is non-positive.')

    @property
    def best_acc(self):
        return self.snapshot_manager.best_acc

    @property
    def best_epoch(self):
        return self.snapshot_manager.best_epoch

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
        was_training = model.training
        self.snapshot_manager.record_selected_points(epoch, selected_indexes)

        if not self.snapshot_manager.should_log(total_step=total_step, batch_idx=batch_idx, force=force):
            return {
                'logged': False,
                'is_best': False,
                'best_acc': self.best_acc,
                'best_epoch': self.best_epoch,
            }

        self.snapshot_manager.mark_logged(total_step)
        model.eval()

        log_data = {
            'diagnostics/trigger': trigger,
            'diagnostics/total_step': total_step,
        }
        log_data['diagnostics/epoch'] = int(epoch)

        snapshot_metrics = None
        if self.should_build_snapshot:
            snapshot, snapshot_metrics = self.snapshot_manager.build_snapshot(model, device, total_step, epoch)
            self.snapshot_manager.store_snapshot(snapshot, total_step)
            is_best = self.snapshot_manager.update_best_checkpoint(epoch, snapshot_metrics, checkpoint_state)
            self.snapshot_manager.log_summary(epoch, total_step, lr, total_time, time_this_epoch, snapshot_metrics)

            if self.wandb_loss_acc:
                log_data['train_loss'] = snapshot_metrics['train_loss']
                log_data['train_acc'] = snapshot_metrics['train_acc']
                log_data['train_loss_train_loader_labels'] = snapshot_metrics['train_loss']
                if self.snapshot_manager.uses_true_labels_for_train_accuracy():
                    log_data['train_acc_true_labels'] = snapshot_metrics['train_acc']
                else:
                    log_data['train_acc_train_loader_labels'] = snapshot_metrics['train_acc']
                log_data['val_loss'] = snapshot_metrics['val_loss']
                log_data['val_acc'] = snapshot_metrics['val_acc']
                log_data['best_val_acc'] = float(self.best_acc)
                log_data['epoch'] = int(epoch)
                log_data['lr'] = float(lr)
                log_data['total_time'] = float(total_time)
                log_data["total_time_str"] = str(timedelta(seconds=int(total_time)))
                log_data['time_epoch'] = float(time_this_epoch)

            if self.wandb_normed_logits:
                log_data['diagnostics/fixed_train_logits_norm_l2_mean'] = snapshot_metrics['train_normed_logits_l2_mean']
                log_data['diagnostics/test_logits_norm_l2_mean'] = snapshot_metrics['val_normed_logits_l2_mean']
        else:
            is_best = False

        percent_noisy_selected = self.snapshot_manager.get_percent_noisy_selected()
        if percent_noisy_selected is not None:
            log_data['percent noisy points selected'] = float(percent_noisy_selected)

        if self.should_log_probe:
            log_data.update(self.probe_diagnostics.log_metrics(model, device))

        if self.should_log_ntk:
            log_data.update(self.ntk_diagnostics.log_metrics(model, device, total_step=total_step))
        
        # TEMP: log norms of weights
        log_data.update()

        self.logger.wandb_log(log_data, step=int(total_step))

        if was_training:
            model.train()

        return {
            'logged': True,
            'is_best': is_best,
            'best_acc': self.best_acc,
            'best_epoch': self.best_epoch,
        }

    def log_step_param_grad_stats(self, model, total_step, epoch):
        if not self.should_log_param_stats:
            return

        log_data = self.param_grad_diagnostics.log_metrics(model)
        if not log_data:
            return

        log_data['diagnostics/trigger'] = 'batch_update_param_grad'
        log_data['diagnostics/total_step'] = int(total_step)
        log_data['diagnostics/epoch'] = int(epoch)
        self.logger.wandb_log(log_data, step=int(total_step))

    def log_epoch_end_selection_stats(self, epoch, total_step):
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
                'percent noisy points selected': noisy_selection_stats['fraction_of_train'],
                'fraction noisy points selected': noisy_selection_stats['fraction_of_noisy_pool'],
            },
            step=int(total_step),
        )

    def finalize(self):
        self.ntk_diagnostics.finalize()
        self.snapshot_manager.finalize()
