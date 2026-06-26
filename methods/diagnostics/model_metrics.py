"""Model-direct diagnostics (§5.8.4, Phase 5c): parameter/gradient norms,
weight-matrix norms, and the linear probe. Each is a single logged leaf that
reuses the existing compute module unchanged (same math, smaller unit). These
read the model (and, for the probe, the loaders) directly and share no
computation with other diagnostics, so no dependency layer is introduced.
"""

from methods.diagnostics.base import Diagnostic, DiagnosticInfo
from methods.method_utils.param_grad import ParamGradDiagnostics
from methods.method_utils.probe import ProbeDiagnostics
from methods.method_utils.weight_matrix import WeightMatrixDiagnostics


class ParamNorms(Diagnostic):
    """L2 norm of all trainable parameters."""

    def __init__(self, manager, builder, context, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        self._impl = ParamGradDiagnostics(wandb_param_norms=True, wandb_grad_norms=False, logger=context.logger)

    def _run(self):
        return DiagnosticInfo("param_norms", self._impl.log_metrics(self.get_context()["model"]))

    def __eq__(self, other):
        return isinstance(other, ParamNorms)


class GradNorms(Diagnostic):
    """L2 norm of the current minibatch gradients."""

    def __init__(self, manager, builder, context, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        self._impl = ParamGradDiagnostics(wandb_param_norms=False, wandb_grad_norms=True, logger=context.logger)

    def _run(self):
        return DiagnosticInfo("grad_norms", self._impl.log_metrics(self.get_context()["model"]))

    def __eq__(self, other):
        return isinstance(other, GradNorms)


class WeightMatrixNorms(Diagnostic):
    """Frobenius / spectral / alignment norms of selected 2D weight matrices."""

    def __init__(self, manager, builder, context, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        last_n = params.get("last_n_layers")
        self._impl = WeightMatrixDiagnostics(
            logger=context.logger,
            context=context,
            enabled=True,
            param_names=params.get("param_names"),
            last_n_layers=int(last_n) if last_n is not None else None,
        )

    def _run(self):
        return DiagnosticInfo("weight_matrix_norms", self._impl.log_metrics(self.get_context()["model"]))

    def __eq__(self, other):
        return isinstance(other, WeightMatrixNorms)


class LinearProbe(Diagnostic):
    """Train/test accuracy of a linear classifier fit on penultimate features."""

    def __init__(self, manager, builder, context, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        self._impl = ProbeDiagnostics(
            logger=context.logger,
            context=context,
            lr_max_iter=int(params.get("max_iter", 300)),
            lr_max_samples=int(params.get("max_samples", -1)),
            enabled=True,
        )

    def _run(self):
        ctx = self.get_context()
        return DiagnosticInfo("linear_probe", self._impl.log_metrics(ctx["model"], ctx["device"]))

    def __eq__(self, other):
        return isinstance(other, LinearProbe)
