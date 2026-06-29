"""Model-direct diagnostics: parameter/gradient norms,
weight-matrix norms, and the linear probe. Holds the compute engines (moved here
from the deleted ``method_utils/{param_grad,probe,weight_matrix}.py``) plus the
logged leaves. Each leaf builds its engine from the manager's static context at
construction and reads ``model``/``device`` from the merged context at run time.
These diagnostics share no computation, so there is no dependency layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.diagnostics.base import Diagnostic, DiagnosticInfo


# --------------------------------------------------------------------------- #
# Compute engines
# --------------------------------------------------------------------------- #

class ParamGradDiagnostics:
    def __init__(self, wandb_param_norms, wandb_grad_norms, logger=None):
        self.wandb_param_norms = bool(wandb_param_norms)
        self.wandb_grad_norms = bool(wandb_grad_norms)
        self.enabled = self.wandb_param_norms or self.wandb_grad_norms
        self.logger = logger

    def log_metrics(self, model):
        """Gets parameters and gradients with respect to parameters in chunks, then logs L2 norms"""
        if not self.enabled:
            return {}

        log_data = {}
        param_chunks = []
        grad_chunks = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            param_cpu = param.detach().float().reshape(-1).cpu()
            if self.wandb_param_norms and param_cpu.numel() > 0:
                param_chunks.append(param_cpu)

            if self.wandb_grad_norms and param.grad is not None:
                grad_cpu = param.grad.detach().float().reshape(-1).cpu()
                if grad_cpu.numel() > 0:
                    grad_chunks.append(grad_cpu)

        if self.wandb_param_norms and param_chunks:
            flat_params = torch.cat(param_chunks, dim=0)
            log_data['diagnostics/parameter_norm_l2'] = torch.norm(flat_params, p=2).item()

        if self.wandb_grad_norms and grad_chunks:
            flat_grads = torch.cat(grad_chunks, dim=0)
            log_data['diagnostics/gradient_norm_l2_minibatch'] = torch.norm(flat_grads, p=2).item()

        return log_data


class WeightMatrixDiagnostics:
    def __init__(self, logger, enabled=False, param_names=None, last_n_layers=None):
        self.logger = logger
        self.enabled = enabled
        self.param_names = param_names
        self.last_n_layers = last_n_layers

    def _get_weight_info(self, name, p):
        if len(p.shape) != 2:
            return None

        frobenius = torch.linalg.norm(p, ord='fro').detach().cpu().item()
        spectral = torch.linalg.matrix_norm(p, ord=2).detach().cpu().item()
        alignment = spectral / frobenius

        return {'frobenius': frobenius, 'spectral': spectral, 'alignment': alignment}

    def log_metrics(self, model):
        if not self.enabled:
            return {}

        all_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        matrix_params = [(n, p) for n, p in all_params if len(p.shape) == 2]

        if self.param_names is not None:
            name_set = set(self.param_names)
            params = [(n, p) for n, p in all_params if n in name_set]
        elif self.last_n_layers is not None:
            if self.last_n_layers > len(matrix_params):
                self.logger.info(
                    f'Warning: weight_matrix_last_n_layers={self.last_n_layers} exceeds the number of '
                    f'2D weight matrices ({len(matrix_params)}); using all.'
                )
            params = matrix_params[-self.last_n_layers:]
        else:
            params = matrix_params

        log_data = {}
        for name, p in params:
            info = self._get_weight_info(name, p)
            if info is None:
                continue
            for metric, value in info.items():
                log_data[f'diagnostics/weight_norms/{name}/{metric}'] = value

        return log_data


class ProbeDiagnostics:
    def __init__(self, logger, train_loader, test_loader, lr_max_iter, lr_max_samples, enabled=True):
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_max_iter = int(lr_max_iter)
        self.lr_max_samples = int(lr_max_samples)
        self.enabled = bool(enabled)
        self._reported_probe_sample_caps = set()
        self._warned_missing_test_loader = False
        self._warned_feature_fallback = False

    def _extract_penultimate_features(self, model, inputs):
        """Returns penultimate layer feature vectors for a batch of inputs"""
        # Prefer explicit feature API; fall back to need_features=True for wrappers.
        with torch.no_grad():
            if hasattr(model, 'feat_nograd_forward'):
                outputs = model.feat_nograd_forward(inputs)
            elif hasattr(model, 'net') and hasattr(model.net, 'feat_nograd_forward'):
                outputs = model.net.feat_nograd_forward(inputs)
            else:
                outputs = model(inputs, need_features=True)

        features = outputs[1]

        return features

    def _collect_penultimate_features(self, model, device, loader, max_samples=-1):
        """Returns penultimate layer feature vectors for all points in dataloader"""
        features_list = []
        labels_list = []
        num_collected = 0

        with torch.no_grad():
            for datas in loader:
                if max_samples > 0 and num_collected >= max_samples:
                    break

                inputs = datas['input'].to(device)
                targets = datas['target'].cpu().long()
                if max_samples > 0:
                    remaining = max_samples - num_collected
                    if remaining <= 0:
                        break
                    if inputs.size(0) > remaining:
                        inputs = inputs[:remaining]
                        targets = targets[:remaining]

                try:
                    features = self._extract_penultimate_features(model, inputs)
                except Exception as exc:
                    if not self._warned_feature_fallback:
                        self.logger.info(
                            f'Warning: failed to extract penultimate features for linear probing ({exc}); skipping probe logging.'
                        )
                        self._warned_feature_fallback = True
                    return None, None

                if features is None:
                    if not self._warned_feature_fallback:
                        self.logger.info(
                            'Warning: linear probe diagnostics require feat_nograd_forward or need_features=True support; '
                            'skipping probe logging.'
                        )
                        self._warned_feature_fallback = True
                    return None, None

                features = features.detach().reshape(features.size(0), -1).cpu()
                features_list.append(features)
                labels_list.append(targets)
                num_collected += targets.size(0)

        if not features_list:
            return None, None

        return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)

    def log_metrics(self, model, device):
        """Creates penultimate layer feature matrix for each dataset,
        runs linear regression on the feature matrix of the training set,
        and returns the accuracy of the linear probe on the feature matrices
        of the training and test sets."""
        if not self.enabled:
            return {}

        if self.test_loader is None:
            if not self._warned_missing_test_loader:
                self.logger.info(
                    'Warning: linear probe diagnostics require a test loader for evaluation; skipping probe logging.'
                )
                self._warned_missing_test_loader = True
            return {}

        if self.lr_max_samples > 0:
            for split_name in ('train', 'test'):
                if split_name not in self._reported_probe_sample_caps:
                    self.logger.info(
                        f"Linear probe will use at most {self.lr_max_samples} {split_name} samples per checkpoint."
                    )
                    self._reported_probe_sample_caps.add(split_name)

        train_features, train_labels = self._collect_penultimate_features(
            model,
            device,
            self.train_loader,
            max_samples=self.lr_max_samples,
        )
        if train_features is None or train_labels is None:
            return {}

        if train_features.shape[0] < 2 or torch.unique(train_labels).numel() < 2:
            return {}

        test_features, test_labels = self._collect_penultimate_features(
            model,
            device,
            self.test_loader,
            max_samples=self.lr_max_samples,
        )
        if test_features is None or test_labels is None:
            return {}

        if test_features.shape[1] != train_features.shape[1]:
            raise RuntimeError(
                'LR probe feature dimension mismatch for penultimate features: '
                f'train dim={train_features.shape[1]}, test dim={test_features.shape[1]}.'
            )

        classifier, feat_mean, feat_std, fit_device = self._fit_gpu_linear_probe(
            train_features,
            train_labels,
            max_iter=self.lr_max_iter,
        )

        train_preds = self._predict_gpu_linear_probe(
            classifier,
            feat_mean,
            feat_std,
            train_features,
            fit_device,
        )

        test_preds = self._predict_gpu_linear_probe(
            classifier,
            feat_mean,
            feat_std,
            test_features,
            fit_device,
        )
        return {
            'diagnostics/penultimate_layer_lr/train_acc': float(
                (train_preds == train_labels).float().mean().item()
            ),
            'diagnostics/penultimate_layer_lr/test_acc': float(
                (test_preds == test_labels).float().mean().item()
            )
        }

    def _fit_gpu_linear_probe(self, train_features, train_labels, max_iter):
        """Fits a linear classifier (probe) to the penultimate layer feature matrix"""
        fit_device = torch.device('cuda' if torch.cuda.is_available() else train_features.device)
        x = train_features.to(device=fit_device, dtype=torch.float32, non_blocking=True)
        y = train_labels.to(device=fit_device, dtype=torch.long, non_blocking=True)

        feat_mean = x.mean(dim=0, keepdim=True)
        feat_std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        x = (x - feat_mean) / feat_std

        num_classes = int(y.max().item()) + 1
        classifier = nn.Linear(x.size(1), num_classes, bias=True).to(fit_device)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-3, weight_decay=1e-4)

        batch_size = min(4096, x.size(0))
        num_epochs = max(8, min(60, int(max_iter) // 5 if int(max_iter) > 0 else 20))
        best_loss = float('inf')
        stale_epochs = 0
        patience = 6

        for _ in range(num_epochs):
            permutation = torch.randperm(x.size(0), device=fit_device)
            epoch_loss = 0.0
            for start in range(0, x.size(0), batch_size):
                batch_indices = permutation[start : start + batch_size]
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]
                logits = classifier(batch_x)
                loss = F.cross_entropy(logits, batch_y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * batch_x.size(0)

            epoch_loss /= float(x.size(0))
            if epoch_loss + 1e-5 < best_loss:
                best_loss = epoch_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        classifier.eval()
        return classifier, feat_mean, feat_std, fit_device

    @staticmethod
    def _predict_gpu_linear_probe(classifier, feat_mean, feat_std, features, fit_device):
        """Evaluates linear probe on a given feature matrix"""
        x = features.to(device=fit_device, dtype=torch.float32, non_blocking=True)
        x = (x - feat_mean) / feat_std
        with torch.no_grad():
            logits = classifier(x)
            preds = logits.argmax(dim=1)
        return preds.cpu()


# --------------------------------------------------------------------------- #
# Logged leaves
# --------------------------------------------------------------------------- #

class ParamNorms(Diagnostic):
    """L2 norm of all trainable parameters."""

    def __init__(self, manager, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        self._impl = ParamGradDiagnostics(
            wandb_param_norms=True, wandb_grad_norms=False, logger=self.method.logger
        )

    def _run(self):
        return DiagnosticInfo("param_norms", self._impl.log_metrics(self.method.model))

    def __eq__(self, other):
        return isinstance(other, ParamNorms)


class GradNorms(Diagnostic):
    """L2 norm of the current minibatch gradients."""

    def __init__(self, manager, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        self._impl = ParamGradDiagnostics(
            wandb_param_norms=False, wandb_grad_norms=True, logger=self.method.logger
        )

    def _run(self):
        return DiagnosticInfo("grad_norms", self._impl.log_metrics(self.method.model))

    def __eq__(self, other):
        return isinstance(other, GradNorms)


class WeightMatrixNorms(Diagnostic):
    """Frobenius / spectral / alignment norms of selected 2D weight matrices."""

    def __init__(self, manager, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        last_n = params.get("last_n_layers")
        self._impl = WeightMatrixDiagnostics(
            logger=self.method.logger,
            enabled=True,
            param_names=params.get("param_names"),
            last_n_layers=int(last_n) if last_n is not None else None,
        )

    def _run(self):
        return DiagnosticInfo("weight_matrix_norms", self._impl.log_metrics(self.method.model))

    def __eq__(self, other):
        return isinstance(other, WeightMatrixNorms)


class LinearProbe(Diagnostic):
    """Train/test accuracy of a linear classifier fit on penultimate features."""

    def __init__(self, manager, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        self._impl = ProbeDiagnostics(
            logger=self.method.logger,
            train_loader=self.method.fixed_train_loader,
            test_loader=self.method.test_loader,
            lr_max_iter=int(params.get("max_iter", 300)),
            lr_max_samples=int(params.get("max_samples", -1)),
            enabled=True,
        )

    def _run(self):
        model = self.method.model
        return DiagnosticInfo("linear_probe", self._impl.log_metrics(model, next(model.parameters()).device))

    def __eq__(self, other):
        return isinstance(other, LinearProbe)
