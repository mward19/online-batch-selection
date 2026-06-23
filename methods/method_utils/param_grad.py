import torch


class ParamGradDiagnostics:
    def __init__(self, wandb_param_norms, wandb_grad_norms, logger=None):
        self.wandb_param_norms = bool(wandb_param_norms)
        self.wandb_grad_norms = bool(wandb_grad_norms)
        self.enabled = self.wandb_param_norms or self.wandb_grad_norms
        self.logger = logger
        # Kept for backward compatibility with existing constructor calls.

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