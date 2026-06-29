"""Diagnostic registry. Maps config class names to constructors.

Definitions live in standard.py (and model_metrics.py / ntk.py).
"""

from methods.diagnostics.standard import (
    TrainLoss, TrainAcc, ValLoss, ValAcc,
    TrueLabelTrainLoss, TrueLabelTrainAcc,
    LogitNormL2, Progress, Checkpoint, SelectedPoints, Timing,
    MinibatchScores,
)
from methods.diagnostics.model_metrics import GradNorms, LinearProbe, ParamNorms, WeightMatrixNorms
from methods.diagnostics.ntk import NTK

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
    "MinibatchScores": MinibatchScores,
}
EPOCH_END_DIAGNOSTICS = {
    "SelectedPoints": SelectedPoints,
    "Timing": Timing,
}
