import torch
import wandb

from methods.method_utils.diagnostics_context import DiagnosticsRunContext

class WeightDiagnostics:
    def __init__(
        self,
        logger,
        context: DiagnosticsRunContext,
    ):
        self.logger = logger

    def _get_weight_info(self, name, p):
        # TODO: add options to control which weight information to calculate
        if len(p.shape) != 2:
            self.logger.info(f"Warning: parameter {name} was not a matrix, not calculating weight diagnostics")
            return None
    
        log_data = dict()
        log_data["Frobenius Norm"] = torch.linalg.norm(p, ord='fro')
        log_data["Matrix 2-Norm"] = torch.linalg.matrix_norm(p, ord=2)
        
        log_data["Alignment Metric"] = log_data["Matrix 2-Norm"] / log_data["Frobenius Norm"]
        
        return log_data

        
    def log_metrics(self, model, param_names=None):
        if param_names is None:
            # Perform diagnostics on all parameters
            params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        else:
            params = [
                (n, p) for n, p in model.named_parameters()
                if p.requires_grad and n in param_names
            ]
        
        if len(params) == 0:
            raise ValueError("No parameters found on which to calculate weight diagnostics")
        
        log_data = dict()
        for name, p in params:
            log_data[f'diagnostics/param_info/{name}'] = self._get_weight_info(name, p)
            # TODO: expose to upper level
        
        return log_data


