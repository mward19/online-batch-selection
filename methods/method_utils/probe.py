import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.method_utils.diagnostics_context import DiagnosticsRunContext


class ProbeDiagnostics:
    def __init__(self, logger, context: DiagnosticsRunContext, lr_max_iter, lr_max_samples, enabled=True):
        self.logger = logger
        self.train_loader = context.fixed_train_loader
        self.test_loader = context.test_loader
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
