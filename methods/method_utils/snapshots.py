import os

import numpy as np
from scipy import optimize
import torch
import torch.nn.functional as F

from methods.method_utils.diagnostics_context import DiagnosticsRunContext


class SnapshotManager:
    def __init__(
        self,
        logger,
        context: DiagnosticsRunContext,
        should_track_points,
        should_save_snapshots,
        should_save_best_checkpoint,
        initial_best_acc=0.0,
        initial_best_epoch=0,
        checkpoint_saver=None,
    ):
        self.logger = logger
        self.context = context
        self.fixed_train_loader = context.fixed_train_loader
        self.test_loader = context.test_loader
        self.total_batches = int(context.total_batches)
        self.num_train_samples = int(context.num_train_samples)
        self.num_epochs = context.num_epochs
        self.num_steps = context.num_steps
        self.should_track_points = should_track_points
        self.should_save_snapshots = should_save_snapshots
        self.should_save_best_checkpoint = should_save_best_checkpoint
        self.wandb_progress = bool(getattr(context, 'wandb_progress', False))
        self.best_acc = float(initial_best_acc if initial_best_acc is not None else context.initial_best_acc)
        self.best_epoch = int(initial_best_epoch if initial_best_epoch is not None else context.initial_best_epoch)
        self.checkpoint_saver = checkpoint_saver if checkpoint_saver is not None else context.checkpoint_saver
        self.true_labels = self._to_long_tensor(context.true_labels)
        if self.true_labels is not None and self.true_labels.numel() != self.num_train_samples:
            raise ValueError(
                f'true_labels has length {self.true_labels.numel()}, expected {self.num_train_samples}.'
            )
        self.noisy_indices = self._to_numpy_indices(context.noisy_indices)
        self.current_epoch = None
        self.current_epoch_selected_mask = None
        if self.noisy_indices is not None and self.noisy_indices.size > 0:
            self.current_epoch_selected_mask = np.zeros(self.num_train_samples, dtype=np.uint8)

        self.snapshots = []
        self.snapshot_steps = []
        snapshots_dir = os.path.join(self.context.project_root, 'snapshots', self.context.dataset_name)
        os.makedirs(snapshots_dir, exist_ok=True)
        self.snapshots_path = os.path.join(snapshots_dir, f'{self.context.artifact_stem}.p')

        self.selected_points_by_epoch = None
        self.selected_points_path = None
        if self.should_track_points:
            self._initialize_selected_points_tracking()

    @staticmethod
    def _to_long_tensor(values):
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().long().reshape(-1)
        return torch.as_tensor(values, dtype=torch.long).reshape(-1)

    @staticmethod
    def _to_numpy_indices(values):
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        return np.asarray(values, dtype=np.int64).reshape(-1)

    def _initialize_selected_points_tracking(self):
        """Initializes self.selected_points_by_epoch: shape (num_epochs, num_train_samples)"""
        num_tracking_epochs = self.num_epochs or int(np.ceil(self.num_steps / self.total_batches))
        self.selected_points_by_epoch = np.zeros(
            (num_tracking_epochs, self.num_train_samples),
            dtype=np.uint8,
        )
        selected_points_dir = os.path.join(self.context.project_root, 'selected_points', self.context.dataset_name)
        os.makedirs(selected_points_dir, exist_ok=True)
        self.selected_points_path = os.path.join(
            selected_points_dir,
            f'{self.context.artifact_stem}.npy',
        )

    def record_selected_points(self, epoch, indexes):
        """Records points selected inside of self.selected_points_by_epoch"""
        """Called after every mini-batch update (after_batch in SelectionMethod.py)"""
        if indexes is None:
            return

        epoch_idx = epoch - 1
        if isinstance(indexes, torch.Tensor):
            selected_indexes = indexes.detach().cpu().numpy()
        else:
            selected_indexes = np.asarray(indexes)

        selected_indexes = np.asarray(selected_indexes, dtype=np.int64).reshape(-1)
        if selected_indexes.size == 0:
            return

        valid_mask = (selected_indexes >= 0) & (selected_indexes < self.num_train_samples)
        selected_indexes = selected_indexes[valid_mask]
        if selected_indexes.size == 0:
            return

        if self.should_track_points:
            self.selected_points_by_epoch[epoch_idx, selected_indexes] = 1

        if self.current_epoch_selected_mask is not None:
            if self.current_epoch != int(epoch):
                self.current_epoch_selected_mask.fill(0)
                self.current_epoch = int(epoch)
            self.current_epoch_selected_mask[selected_indexes] = 1

    def get_noisy_selection_stats(self):
        """Calculate number of noisy points selected by epoch"""
        """Called in Diagnostics.log_epoch_end_selection_stats() function"""
        if self.current_epoch_selected_mask is None:
            return None

        num_noisy_selected = int(self.current_epoch_selected_mask[self.noisy_indices].sum())
        fraction_of_train = float(num_noisy_selected / self.num_train_samples)
        total_noisy = int(self.noisy_indices.size)
        fraction_of_noisy_pool = float(num_noisy_selected / total_noisy) if total_noisy > 0 else 0.0
        return {
            'num_noisy_selected': num_noisy_selected,
            'fraction_of_train': fraction_of_train,
            'fraction_of_noisy_pool': fraction_of_noisy_pool,
            'total_noisy': total_noisy,
        }

    def has_label_noise(self):
        """Used to determine if our data has label noise"""
        return self.true_labels is not None

    def _calculate_snapshot_stats(self, model, loader, device, override_labels=None):
        """Calculate per-sample log-probabilities, loss, error, and logits norms."""
        log_probs = []
        logits_l2_norms = []
        losses = []
        errors = []
        eval_labels = []
        noisy_losses = []
        noisy_errors = []
        has_noisy_metrics = override_labels is not None
        # Override labels with true labels if we want to get accuracy
        for datas in loader:
            inputs = datas['input'].to(device)
            targets = datas['target'].to(device)
            batch_logits = model(inputs)
            batch_log_probs = F.log_softmax(batch_logits, dim=1)
            predicted = torch.argmax(batch_logits, dim=1).long()
            batch_logits_l2 = torch.norm(batch_logits, p=2, dim=1)
            eval_targets = targets
            if override_labels is not None:
                indexes = datas['index']
                if not isinstance(indexes, torch.Tensor):
                    indexes = torch.as_tensor(indexes, dtype=torch.long)
                eval_targets = override_labels[indexes.detach().cpu()].to(device)
            batch_losses = -torch.gather(batch_log_probs, 1, eval_targets.view(-1, 1))
            batch_errors = (eval_targets != predicted).float()
            eval_labels.append(eval_targets.detach().cpu().long())
            if has_noisy_metrics:
                batch_noisy_losses = -torch.gather(batch_log_probs, 1, targets.view(-1, 1))
                batch_noisy_errors = (targets != predicted).float()
                noisy_losses.append(batch_noisy_losses.detach().cpu())
                noisy_errors.append(batch_noisy_errors.detach().cpu())
            log_probs.append(batch_log_probs.detach().cpu())
            logits_l2_norms.append(batch_logits_l2.detach().cpu())
            losses.append(batch_losses.detach().cpu())
            errors.append(batch_errors.detach().cpu())

        noisy_losses_tensor = torch.cat(noisy_losses, dim=0) if has_noisy_metrics else None
        noisy_errors_tensor = torch.cat(noisy_errors, dim=0) if has_noisy_metrics else None

        return (
            torch.cat(log_probs, dim=0),
            torch.cat(logits_l2_norms, dim=0),
            torch.cat(losses, dim=0),
            torch.cat(errors, dim=0),
            torch.cat(eval_labels, dim=0),
            noisy_losses_tensor,
            noisy_errors_tensor,
        )

    def _compute_progress(self, log_probs, labels):
        """Compute geodesic progress between uniform predictions and one-hot labels."""
        if log_probs.numel() == 0:
            return 0.0

        probabilities = torch.exp(log_probs.detach().cpu())
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
        predictions_on_sphere = np.sqrt(probabilities.numpy().astype(np.float64, copy=False))

        label_indices = labels.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        num_samples, num_classes = predictions_on_sphere.shape
        if label_indices.size != num_samples:
            raise ValueError(f'labels length {label_indices.size} does not match predictions {num_samples}.')

        valid_labels = (label_indices >= 0) & (label_indices < num_classes)
        if not np.all(valid_labels):
            raise ValueError('Found label outside valid class range when computing progress.')

        one_hot_labels = np.zeros((num_samples, num_classes), dtype=np.float64)
        one_hot_labels[np.arange(num_samples), label_indices] = 1.0
        ground_truth = np.sqrt(one_hot_labels)
        ignorance = np.sqrt(np.full((num_samples, num_classes), 1.0 / float(num_classes), dtype=np.float64))

        eps = 1e-8
        ignorance_ground_truth_cosine = np.clip((ignorance * ground_truth).sum(axis=1), 0.0, 1.0)
        ignorance_predictions_cosine = np.clip((ignorance * predictions_on_sphere).sum(axis=1), 0.0, 1.0)
        ground_truth_predictions_cosine = np.clip((ground_truth * predictions_on_sphere).sum(axis=1), 0.0, 1.0)
        ignorance_ground_truth_angle = np.arccos(ignorance_ground_truth_cosine)
        degenerate_mask = ignorance_ground_truth_angle < eps

        degenerate_distance = float(np.arccos(ignorance_predictions_cosine[degenerate_mask]).sum()) if np.any(degenerate_mask) else 0.0
        if np.all(degenerate_mask):
            return 0.0

        non_degenerate_angle = ignorance_ground_truth_angle[~degenerate_mask]
        non_degenerate_ignorance_predictions_cosine = ignorance_predictions_cosine[~degenerate_mask]
        non_degenerate_ground_truth_predictions_cosine = ground_truth_predictions_cosine[~degenerate_mask]
        non_degenerate_angle_sin = np.sin(non_degenerate_angle)

        def objective_fn(t):
            geodesic_cosine = (
                non_degenerate_ignorance_predictions_cosine * np.sin((1.0 - t) * non_degenerate_angle) / non_degenerate_angle_sin
                + non_degenerate_ground_truth_predictions_cosine * np.sin(t * non_degenerate_angle) / non_degenerate_angle_sin
            )
            geodesic_cosine = np.clip(geodesic_cosine, 0.0, 1.0)
            return degenerate_distance + float(np.arccos(geodesic_cosine).sum())

        lam = optimize.minimize_scalar(objective_fn, bounds=(0.0, 1.0), method='bounded').x
        return float(np.clip(lam, 0.0, 1.0))

    def build_snapshot(self, model, device, total_step, epoch):
        """Build metrics for logging to wandb and snapshots for saving locally (used to compute progress)"""
        with torch.no_grad():
            train_log_probs, train_logits_l2_norms, train_losses, train_errors, train_eval_labels, train_losses_noisy, train_errors_noisy = self._calculate_snapshot_stats(
                model,
                self.fixed_train_loader,
                device,
<<<<<<< HEAD
=======
                override_labels=self.true_labels,
>>>>>>> main
            )
            test_log_probs, test_logits_l2_norms, test_losses, test_errors, test_eval_labels, _, _ = self._calculate_snapshot_stats(model, self.test_loader, device)

        metrics = {
            'train_loss': float(train_losses.mean().item()),
            'train_acc': float(1.0 - train_errors.mean().item()),
            'val_loss': float(test_losses.mean().item()),
            'val_acc': float(1.0 - test_errors.mean().item()),
            'train_normed_logits_l2_mean': float(train_logits_l2_norms.mean().item()),
            'val_normed_logits_l2_mean': float(test_logits_l2_norms.mean().item()),
        }
        if train_losses_noisy is not None and train_errors_noisy is not None:
            metrics['train_loss_noisy_labels'] = float(train_losses_noisy.mean().item())
            metrics['train_acc_noisy_labels'] = float(1.0 - train_errors_noisy.mean().item())
        if self.wandb_progress:
            metrics['train_progress'] = self._compute_progress(train_log_probs, train_eval_labels)
            metrics['val_progress'] = self._compute_progress(test_log_probs, test_eval_labels)

        snapshot = {
            'train_log_probs': train_log_probs,
            'train_losses': train_losses,
            'train_errors': train_errors,
            'test_log_probs': test_log_probs,
            'test_losses': test_losses,
            'test_errors': test_errors,
            'total_step': int(total_step),
            'epoch': int(epoch),
        }
<<<<<<< HEAD
        metrics = {
            'train_loss': float(f.mean().item()),
            'train_acc': float(1.0 - e.mean().item()),
            'val_loss': float(fv.mean().item()),
            'val_acc': float(1.0 - ev.mean().item()),
            'train_normed_logits_l2_mean': float(torch.norm(yh, p=2, dim=1).mean().item()),
            'val_normed_logits_l2_mean': float(torch.norm(yvh, p=2, dim=1).mean().item()),
        }
        if self.true_labels is not None:
            yht, ft, et = self._calculate_snapshot_stats(
                model,
                self.fixed_train_loader,
                device,
                true_labels=self.true_labels,
            )
            snapshot.update({
                'yht': yht,
                'ft': ft,
                'et': et,
            })
            metrics.update({
                'train_loss_true_labels': float(ft.mean().item()),
                'train_acc_true_labels': float(1.0 - et.mean().item()),
            })

=======
>>>>>>> main
        return snapshot, metrics

    def store_snapshot(self, snapshot, total_step):
        """Stores current model state snapshot in list self.snapshots"""
        if not self.should_save_snapshots:
            return
        self.snapshots.append(snapshot)
        self.snapshot_steps.append(int(total_step))

    def log_summary(self, epoch, total_step, lr, total_time, time_this_epoch, metrics):
        """log summary statistics to logger"""
        total_epoch_display = self.num_epochs or '?'
        self.logger.info(
            f'=====> Epoch: {epoch}/{total_epoch_display}, Total_step: {total_step}, '
            f'lr: {lr:.6f}, Total Time: {total_time:.4f} s, '
            f'Time this epoch: {time_this_epoch:.4f} s'
        )
        self.logger.info(
            f'=====> Epoch: {epoch}/{total_epoch_display}, '
            f'Train Loss: {metrics["train_loss"]:.4f}, Train acc: {metrics["train_acc"]:.4f}, '
            f'Best val acc: {self.best_acc:.4f}, Current val acc: {metrics["val_acc"]:.4f}, '
            f'Val Loss: {metrics["val_loss"]:.4f}'
        )

    def update_best_checkpoint(self, epoch, metrics, checkpoint_state):
        """If test accuracy increases, save model as best model"""
        is_best = False
        if metrics['val_acc'] > self.best_acc:
            self.best_acc = metrics['val_acc']
            self.best_epoch = int(epoch)
            is_best = True

        if not is_best:
            return False

        if self.should_save_best_checkpoint and self.checkpoint_saver is not None:
            checkpoint_state = dict(checkpoint_state)
            checkpoint_state['best_acc'] = self.best_acc
            checkpoint_state['best_epoch'] = self.best_epoch
            self.checkpoint_saver(checkpoint_state, True)
        return True

    def _save_selected_points(self):
        """Save self.selected_points_by_epoch to numpy array"""
        np.save(self.selected_points_path, self.selected_points_by_epoch)
        self.logger.info(f'Saved selected points array to {self.selected_points_path}')

    def _save_snapshots(self):
        """Save self.snapshots and self.snapshot_steps at the end of the run"""
        payload = {
            'data': self.snapshots,
            'steps': self.snapshot_steps,
        }
        torch.save(payload, self.snapshots_path)
        self.logger.info(f'Saved helper snapshots to {self.snapshots_path}')

    def finalize(self):
        if self.should_save_snapshots:
            self._save_snapshots()
        if self.should_track_points:
            self._save_selected_points()