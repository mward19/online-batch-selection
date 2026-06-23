import numpy as np
import torch

from methods.SelectionMethod import MinibatchInfo
from methods.SelectionMethod import SelectionMethod


class GradNormIS(SelectionMethod):
    method_name = "GradNormIS"

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.selection_method = "GradNormIS"
        self.balance = config["method_opt"]["balance"]
        self.ratio = config["method_opt"]["ratio"]
        self.ratio_scheduler = (
            config["method_opt"]["ratio_scheduler"]
            if "ratio_scheduler" in config["method_opt"]
            else "constant"
        )
        self.warmup_epochs = (
            config["method_opt"]["warmup_epochs"]
            if "warmup_epochs" in config["method_opt"]
            else 0
        )
        self.reduce_dim = (
            config["method_opt"]["reduce_dim"]
            if "reduce_dim" in config["method_opt"]
            else False
        )

        # Sample with replacement
        self.replacement = self.config["method_opt"]["sample_with_replacement"]

        self.uniform_fallback = self.config["method_opt"]["uniform_fallback"]

        if self.uniform_fallback:
            self.tau = 0
            self.a_tau = self.config["method_opt"]["uniform_fallback_opt"]["a_tau"]

        self.probability_threshold = self.config["method_opt"].get("probability_threshold", 0.0)

    def get_ratio_per_epoch(self, epoch):
        if epoch < self.warmup_epochs:
            self.logger.info("warming up")
            return 1.0
        if self.ratio_scheduler == "constant":
            return self.ratio
        elif self.ratio_scheduler == "increase_linear":
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == "decrease_linear":
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == "increase_exp":
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * np.exp(epoch / self.epochs)
        elif self.ratio_scheduler == "decrease_exp":
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * np.exp(epoch / self.epochs)
        else:
            raise NotImplementedError

    def calc_grad(self, inputs, targets, indexes):
        model = (
            self.model.module
            if isinstance(self.model, torch.nn.DataParallel)
            else self.model
        )
        model.eval()
        outputs, features = model.feat_nograd_forward(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        with torch.no_grad():
            grad_out = torch.autograd.grad(loss, outputs, retain_graph=True)[0]
            grad = grad_out.unsqueeze(-1) * features.unsqueeze(1)
            grad = grad.view(grad.shape[0], -1)
        model.train()
        if self.reduce_dim:
            dim = grad.shape[1]
            dim_reduced = dim // self.reduce_dim
            index = np.random.choice(dim, dim_reduced, replace=False)
            grad = grad[:, index]
        grad_mean = grad.mean(dim=0)
        return grad_mean, grad
    
    def normalize_and_threshold(self, weights, min_num_nonzero):
        # Normalize weights so they sum to 1 (turn them into a probability distribution)
        new_weights = weights / torch.sum(weights)
        
        # Apply probability threshold
        mask_above_threshold = new_weights >= self.probability_threshold
        # Make sure at least min_num_nonzero samples have nonzero probability. 
        # Otherwise, don't threshold at all
        if mask_above_threshold.sum() >= min_num_nonzero: 
            new_weights[~mask_above_threshold] = 0.0
        
        # Renormalize
        return new_weights / torch.sum(new_weights)

    def before_batch(self, i, inputs, targets, indexes, epoch):
        ratio = self.get_ratio_per_epoch(epoch)
        if ratio == 1.0:
            if i == 0:
                self.logger.info("using all samples")
            return super().before_batch(i, inputs, targets, indexes, epoch)
        else:
            if i == 0:
                self.logger.info(f"balance: {self.balance}")
                self.logger.info(
                    "selecting samples for epoch {}, ratio {}".format(epoch, ratio)
                )
        _, grad = self.calc_grad(inputs, targets, indexes)
        grad_norm = torch.norm(grad, dim=1)

        B = len(targets) # Same as self.batch_size most of the time
        number_to_select = int(inputs.shape[0] * ratio)
        number_to_select = max(1, min(number_to_select, B)) # clip between 1 and batch size


        if self.uniform_fallback:
            # See discussion at the beginning of Sec. 3.3 of https://arxiv.org/pdf/1803.00942
            b = number_to_select
            tau_th = (B+3*b) / (3*b) 
        
        # Algorithm 1 in https://arxiv.org/pdf/1803.00942
        if not self.uniform_fallback or self.tau > tau_th:
            # Apply a probability threshold for stability
            probabilities = self.normalize_and_threshold(grad_norm, b) 
            minibatch_indices = torch.multinomial(probabilities, number_to_select, replacement=self.replacement)
            weights = (number_to_select * B * probabilities[minibatch_indices])**-1
        else:
            minibatch_indices = torch.randint(B, size=(number_to_select,))
            weights = None # Uniform weighting

        if self.uniform_fallback:
            # See Equation 27 and line 17 of Algorithm 1
            new_tau_inv = 1 - torch.sum((grad_norm - 1/B)**2) / torch.sum(grad_norm**2)
            self.tau = self.a_tau * self.tau + (1 - self.a_tau) * new_tau_inv**-1

        minibatch_indices = minibatch_indices.cpu().numpy()

        inputs = inputs[minibatch_indices]
        targets = targets[minibatch_indices]
        indexes = indexes[minibatch_indices]

        return MinibatchInfo(inputs, targets, indexes, weights)