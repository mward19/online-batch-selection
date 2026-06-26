"""NTK diagnostic (§5.8.4, Phase 5c). A single logged leaf wrapping the existing
``NTKDiagnostics`` compute engine. NTK keeps heavy internal state across steps
(initial kernel, initial top eigenvectors, teacher kernel, spectrum history) and
computes its metrics in one tightly-coupled pass over its own fixed balanced
subset, which nothing else consumes — so a separate ``NTKKernel`` dependency
would share no work and only risk reordering that stateful math. The leaf
therefore delegates to ``NTKDiagnostics.log_metrics`` unchanged.
"""

from methods.diagnostics.base import Diagnostic, DiagnosticInfo
from methods.method_utils.ntk import NTKDiagnostics


class NTK(Diagnostic):
    def __init__(self, manager, builder, context, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        self._impl = NTKDiagnostics(
            logger=context.logger,
            context=context,
            config=context.config,
            num_classes=context.num_classes,
            ntk_max_samples=int(params.get("max_samples", 1000)),
            ntk_top_k=int(params.get("top_k", 10)),
            ntk_variant=str(params.get("variant", params.get("kernel_type", "trace"))),
            ntk_eigenvalue_concentration_checkpoints=params.get(
                "eigenvalue_concentration_checkpoints", [20, 40, 80]
            ),
            enabled=True,
            save_spectrum=bool(params.get("save_spectrum", False)),
        )

    def _run(self):
        ctx = self.get_context()
        total_steps = int(self.get_state().total_steps)
        return DiagnosticInfo("ntk", self._impl.log_metrics(ctx["model"], ctx["device"], total_steps))

    def finalize(self):
        self._impl.finalize()

    def __eq__(self, other):
        return isinstance(other, NTK)
