import torch

from methods.method_utils.diagnostics_context import DiagnosticsRunContext


class WeightMatrixDiagnostics:
    def __init__(
        self,
        logger,
        context: DiagnosticsRunContext,
        enabled: bool = False,
        param_names=None,
        last_n_layers=None,
    ):
        self.logger = logger
        self.context = context
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
