"""Dataset-specific specialty diagnostics.

These log scalar constants exposed by certain dataset loaders via data_info.
They silently no-op on datasets that don't populate the relevant keys.
"""

from methods.diagnostics.base import Diagnostic, DiagnosticInfo, LogType


class WStarTestAcc(Diagnostic):
    """Logs the test accuracy of the Bayes-optimal classifier (w*) as a
    reference line. Value is constant; logging each step keeps it visible
    alongside training curves in W&B. No-ops if the dataset doesn't provide it."""

    def __init__(self, manager, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)

    def _run(self):
        value = self.method.data_info.get("wstar_test_acc")
        if value is None:
            return DiagnosticInfo("wstar_test_acc", {}, log_type=LogType.SUMMARY)
        return DiagnosticInfo("wstar_test_acc", {"wstar_test_acc": value}, log_type=LogType.SUMMARY)

    def __eq__(self, other):
        return isinstance(other, WStarTestAcc)


class WHatTestAcc(Diagnostic):
    """Logs the test accuracy of the noised teacher classifier (w_hat) as a
    reference line. No-ops if the dataset doesn't provide it."""

    def __init__(self, manager, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)

    def _run(self):
        value = self.method.data_info.get("what_test_acc")
        if value is None:
            return DiagnosticInfo("what_test_acc", {}, log_type=LogType.SUMMARY)
        return DiagnosticInfo("what_test_acc", {"what_test_acc": value}, log_type=LogType.SUMMARY)

    def __eq__(self, other):
        return isinstance(other, WHatTestAcc)


class BayesAccAntipodalGaussian(Diagnostic):
    """Analytical Bayes accuracy for antipodal_unit_sphere blobs: Φ(center_scale / cluster_std).
    No-ops on other center types."""

    def __init__(self, manager, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)

    def _run(self):
        dcfg = self.method.config.get('dataset', {})
        if dcfg.get('centers_type') != 'antipodal_unit_sphere':
            return DiagnosticInfo("bayes_acc", {}, log_type=LogType.SUMMARY)
        from scipy.special import ndtr
        center_scale = float(dcfg.get('center_scale', 1.0))
        cluster_std = float(dcfg.get('cluster_std', 1.0))
        value = float(ndtr(center_scale / cluster_std))
        return DiagnosticInfo("bayes_acc", {"bayes_acc": value}, log_type=LogType.SUMMARY)

    def __eq__(self, other):
        return isinstance(other, BayesAccAntipodalGaussian)
