"""Concrete diagnostics (§5.8.4), decomposed into cached compute dependencies
and small logged leaves wired by the §5.3 dependency mechanism.

Layer 1 (compute deps, unregistered, never logged):
  ForwardPass(loader)        -- one model pass over a loader, shared by all metrics
  PerSampleLossError(loader, label_source)
Layer 2 (logged leaves, registered with a manager):
  TrainLoss/TrainAcc/ValLoss/ValAcc, TrueLabel*{Loss,Acc}, LogitNormL2,
  Checkpoint, SelectedPoints
"""

import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from methods.diagnostics.base import Diagnostic, DiagnosticInfo
from run_dir import atomic_save


# --------------------------------------------------------------------------- #
# Layer 1: cached compute dependencies (built via the builder, not registered)
# --------------------------------------------------------------------------- #

class ForwardPass(Diagnostic):
    """One eval pass over ``loader_key`` ('train'|'val'); returns per-sample
    log-probs, logit L2 norms, predictions, loader targets, and indices."""

    def __init__(self, manager, context, loader_key):
        super().__init__(manager)
        self.context = context
        self.loader_key = loader_key
        self.loader = context.fixed_train_loader if loader_key == "train" else context.test_loader

    def _run(self):
        ctx = self.get_context()
        model = ctx["model"]
        device = ctx["device"]
        model.eval()
        log_probs, logit_l2, preds, targets, indices = [], [], [], [], []
        with torch.no_grad():
            for datas in self.loader:
                inputs = datas["input"].to(device)
                tgt = datas["target"].to(device)
                logits = model(inputs)
                log_probs.append(F.log_softmax(logits, dim=1).cpu())
                logit_l2.append(torch.norm(logits, p=2, dim=1).cpu())
                preds.append(torch.argmax(logits, dim=1).long().cpu())
                targets.append(tgt.long().cpu())
                idx = datas["index"]
                idx = idx if isinstance(idx, torch.Tensor) else torch.as_tensor(idx, dtype=torch.long)
                indices.append(idx.long().cpu())
        return DiagnosticInfo(f"forward_{self.loader_key}", {
            "log_probs": torch.cat(log_probs),
            "logit_l2": torch.cat(logit_l2),
            "predictions": torch.cat(preds),
            "targets": torch.cat(targets),
            "indices": torch.cat(indices),
        })

    def __eq__(self, other):
        return isinstance(other, ForwardPass) and self.loader_key == other.loader_key


class PerSampleLossError(Diagnostic):
    """Per-sample loss and 0/1 error for ``loader_key`` against ``label_source``
    ('loader' = the loader's own (possibly noisy) labels, 'true' = clean labels)."""

    def __init__(self, manager, builder, context, loader_key, label_source):
        super().__init__(manager)
        self.loader_key = loader_key
        self.label_source = label_source
        self.forward_pass = builder.build(ForwardPass, manager, context, loader_key)
        self.true_labels = _to_long_vector(context.true_labels)

    def _run(self):
        fp = self.forward_pass.run().info
        log_probs, predictions = fp["log_probs"], fp["predictions"]
        if self.label_source == "true":
            if self.true_labels is None:
                raise ValueError("PerSampleLossError(label_source='true') needs context.true_labels.")
            labels = self.true_labels[fp["indices"]]
        else:
            labels = fp["targets"]
        loss = -torch.gather(log_probs, 1, labels.view(-1, 1)).squeeze(1)
        error = (labels != predictions).float()
        return DiagnosticInfo(f"loss_error_{self.loader_key}_{self.label_source}",
                              {"loss": loss, "error": error})

    def __eq__(self, other):
        return (isinstance(other, PerSampleLossError)
                and self.loader_key == other.loader_key
                and self.label_source == other.label_source)


# --------------------------------------------------------------------------- #
# Layer 2: logged leaves
# --------------------------------------------------------------------------- #

class _LossErrorLeaf(Diagnostic):
    """Shared base for the mean-loss / mean-acc leaves."""
    loader_key = "train"
    label_source = "loader"
    metric = "loss"   # 'loss' or 'acc'
    log_key = "train_loss"

    def __init__(self, manager, builder, context, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        self.dep = builder.build(PerSampleLossError, manager, builder, context,
                                 self.loader_key, self.label_source)

    def _run(self):
        info = self.dep.run().info
        if self.metric == "loss":
            value = float(info["loss"].mean().item())
        else:
            value = float(1.0 - info["error"].mean().item())
        return DiagnosticInfo(self.log_key, {self.log_key: value})


class TrainLoss(_LossErrorLeaf):
    loader_key, label_source, metric, log_key = "train", "loader", "loss", "train_loss"

class TrainAcc(_LossErrorLeaf):
    loader_key, label_source, metric, log_key = "train", "loader", "acc", "train_acc"

class ValLoss(_LossErrorLeaf):
    loader_key, label_source, metric, log_key = "val", "loader", "loss", "val_loss"

class ValAcc(_LossErrorLeaf):
    loader_key, label_source, metric, log_key = "val", "loader", "acc", "val_acc"

class TrueLabelTrainLoss(_LossErrorLeaf):
    loader_key, label_source, metric, log_key = "train", "true", "loss", "train_loss_true_labels"

class TrueLabelTrainAcc(_LossErrorLeaf):
    loader_key, label_source, metric, log_key = "train", "true", "acc", "train_acc_true_labels"


class LogitNormL2(Diagnostic):
    """Mean L2 norm of train/val logits."""

    def __init__(self, manager, builder, context, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        self.train_fp = builder.build(ForwardPass, manager, context, "train")
        self.val_fp = builder.build(ForwardPass, manager, context, "val")

    def _run(self):
        train = float(self.train_fp.run().info["logit_l2"].mean().item())
        val = float(self.val_fp.run().info["logit_l2"].mean().item())
        return DiagnosticInfo("logit_norm_l2", {
            "diagnostics/train_logits_norm_l2_mean": train,
            "diagnostics/test_logits_norm_l2_mean": val,
        })

    def __eq__(self, other):
        return isinstance(other, LogitNormL2)


class Progress(Diagnostic):
    """Geodesic progress (in [0,1]) from the uniform-prediction point toward the
    one-hot labels on the probability sphere, for train and val. Ported from the
    old ``SnapshotManager._compute_progress``; depends on the shared forward
    passes."""

    def __init__(self, manager, builder, context, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        self.train_fp = builder.build(ForwardPass, manager, context, "train")
        self.val_fp = builder.build(ForwardPass, manager, context, "val")

    def _run(self):
        train_fp = self.train_fp.run().info
        val_fp = self.val_fp.run().info
        train = _geodesic_progress(train_fp["log_probs"], train_fp["targets"])
        val = _geodesic_progress(val_fp["log_probs"], val_fp["targets"])
        return DiagnosticInfo("progress", {"train_progress": train, "val_progress": val})

    def __eq__(self, other):
        return isinstance(other, Progress)


class Checkpoint(Diagnostic):
    """Rolling checkpoint + best-model tracking (§9.2/O2). Writes the rolling
    ``snapshots/checkpoint.pth.tar`` atomically every time it fires and copies
    ``model_best.pth.tar`` when val accuracy improves."""

    def __init__(self, manager, builder, context, should_run=None, **params):
        super().__init__(manager, should_run=should_run)
        self.val_acc = builder.build(PerSampleLossError, manager, builder, context, "val", "loader")
        self.snapshots_dir = os.path.join(context.save_dir, "snapshots")
        self.checkpoint_path = os.path.join(self.snapshots_dir, "checkpoint.pth.tar")
        self.best_path = os.path.join(self.snapshots_dir, "model_best.pth.tar")
        self.save_best = bool(params.get("save_best", True))
        self.best_acc = float(context.initial_best_acc or 0.0)
        self.best_epoch = int(context.initial_best_epoch or 0)
        self.is_best = False

    def _run(self):
        state = self.get_state()
        val_acc = float(1.0 - self.val_acc.run().info["error"].mean().item())
        self.is_best = val_acc > self.best_acc
        if self.is_best:
            self.best_acc = val_acc
            self.best_epoch = int(state.epoch)

        checkpoint_state = self.get_context().get("checkpoint_state")
        if checkpoint_state is not None:
            checkpoint_state = dict(checkpoint_state)
            checkpoint_state["best_acc"] = self.best_acc
            checkpoint_state["best_epoch"] = self.best_epoch
            atomic_save(lambda p: torch.save(checkpoint_state, p), self.checkpoint_path)
            if self.is_best and self.save_best:
                shutil.copyfile(self.checkpoint_path, self.best_path)

        return DiagnosticInfo("checkpoint", {"best_val_acc": self.best_acc})

    def __eq__(self, other):
        return isinstance(other, Checkpoint)


class SelectedPoints(Diagnostic):
    """Epoch-end noisy-selection statistics. Reads the epoch's selected-point
    mask from ``shared_context['selected_mask']`` (maintained by the training
    loop) and the noisy indices from context."""

    def __init__(self, manager, builder, context, should_run=None, **params):
        super().__init__(manager, should_run=should_run)
        self.noisy_indices = _to_numpy_indices(context.noisy_indices)
        self.num_train_samples = int(context.num_train_samples)

    def _run(self):
        mask = self.get_context().get("selected_mask")
        if mask is None or self.noisy_indices is None or self.noisy_indices.size == 0:
            return DiagnosticInfo("selected_points", {})
        num_noisy_selected = int(mask[self.noisy_indices].sum())
        total_noisy = int(self.noisy_indices.size)
        return DiagnosticInfo("selected_points", {
            "num noisy points selected": num_noisy_selected,
            "percent of batch with label noise": num_noisy_selected / self.num_train_samples,
            "percent of points with label noise selected":
                (num_noisy_selected / total_noisy) if total_noisy > 0 else 0.0,
        })

    def __eq__(self, other):
        return isinstance(other, SelectedPoints)


from methods.diagnostics.model_metrics import GradNorms, LinearProbe, ParamNorms, WeightMatrixNorms
from methods.diagnostics.ntk import NTK

# Class-name -> constructor, plus which manager phase each leaf belongs to.
POST_BATCH_DIAGNOSTICS = {
    "TrainLoss": TrainLoss,
    "TrainAcc": TrainAcc,
    "ValLoss": ValLoss,
    "ValAcc": ValAcc,
    "TrueLabelTrainLoss": TrueLabelTrainLoss,
    "TrueLabelTrainAcc": TrueLabelTrainAcc,
    "LogitNormL2": LogitNormL2,
    "Progress": Progress,
    "ParamNorms": ParamNorms,
    "GradNorms": GradNorms,
    "WeightMatrixNorms": WeightMatrixNorms,
    "LinearProbe": LinearProbe,
    "NTK": NTK,
    "Checkpoint": Checkpoint,
}
EPOCH_END_DIAGNOSTICS = {
    "SelectedPoints": SelectedPoints,
}


def _geodesic_progress(log_probs, labels):
    """Geodesic interpolation parameter between the uniform (ignorance) point and
    the one-hot ground truth on the unit sphere of sqrt-probabilities."""
    from scipy import optimize

    if log_probs.numel() == 0:
        return 0.0

    probabilities = torch.exp(log_probs.detach().cpu())
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    predictions_on_sphere = np.sqrt(probabilities.numpy().astype(np.float64, copy=False))

    label_indices = labels.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    num_samples, num_classes = predictions_on_sphere.shape
    if label_indices.size != num_samples:
        raise ValueError(f"labels length {label_indices.size} does not match predictions {num_samples}.")

    valid_labels = (label_indices >= 0) & (label_indices < num_classes)
    if not np.all(valid_labels):
        raise ValueError("Found label outside valid class range when computing progress.")

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

    lam = optimize.minimize_scalar(objective_fn, bounds=(0.0, 1.0), method="bounded").x
    return float(np.clip(lam, 0.0, 1.0))


def _to_long_vector(values):
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().long().reshape(-1)
    return torch.as_tensor(values, dtype=torch.long).reshape(-1)


def _to_numpy_indices(values):
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    return np.asarray(values, dtype=np.int64).reshape(-1)
