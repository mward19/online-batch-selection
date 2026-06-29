import numpy as np
import torch
import torch.nn.functional as F

from methods.method_utils.build_teacher_model import build_teacher_model
from methods.method_utils.loss import *
from methods.method_utils.optimizer import *
from methods.SelectionMethod import MinibatchInfo
from methods.SelectionMethod import SelectionMethod


class RhoLoss(SelectionMethod):
    """A class for implementing the RhoLoss selection method, which selects samples based on reducible loss.

    This class inherits from `SelectionMethod` and uses an irreducible loss model (ILmodel) and a target model
    to compute reducible loss for sample selection during training. It supports various ratio scheduling strategies
    for dynamic sample selection and handles model training and loading for specific datasets.

    Args:
        config (dict): Configuration dictionary containing method and dataset parameters.
            Expected keys include:
                - 'method_opt': Dictionary with keys 'ratio', 'budget', 'epochs', 'ratio_scheduler',
                  'warmup_epochs', 'iter_selection', 'balance'.
                - 'rho_loss': Dictionary with key 'training_budget'.
                - 'dataset': Dictionary with keys 'name' and 'num_classes'.
                - 'networks': Dictionary with key 'params' containing 'm_type'.
        logger (logging.Logger): Logger instance for logging training and selection information.
    """
    method_name = 'RhoLoss'
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.balance = config['method_opt']['balance']
        self.ratio = config['method_opt']['ratio']
        self.ratio_scheduler = config['method_opt']['ratio_scheduler'] if 'ratio_scheduler' in config['method_opt'] else 'constant'
        self.warmup_epochs = config['method_opt']['warmup_epochs'] if 'warmup_epochs' in config['method_opt'] else 0
        self.reduce_dim = config['method_opt']['reduce_dim'] if 'reduce_dim' in config['method_opt'] else False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.setup_teacher_model(config, logger)
        self.precompute_losses()
        self.test_teacher() # Validate teacher classifier test accuracy

        # starting with uniform selection generally helps performance
        self.uniform_epochs = config['method_opt']['uniform_epochs'] if 'uniform_epochs' in config['method_opt'] else 0

    def setup_teacher_model(self, config, logger):
        """Retrieve the teacher model from config for computing irreducible loss."""
        teacher_config = dict(config)
        teacher_config['classes'] = self.data_info.get('classes')
        teacher_config['template'] = self.data_info.get('template')
        self.teacher_model = build_teacher_model(teacher_config, logger)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()        
    

    def precompute_losses(self):
        """Precompute irreducible losses for the training dataset using the teacher model."""
        self.teacher_model.eval()
        losses_tensor = torch.zeros(len(self.train_dset))

        with torch.no_grad():
            for datas in self.train_loader:
                inputs = datas['input'].to(self.device)
                targets = datas['target'].to(self.device)
                indexes = datas['index']
                outputs = self.teacher_model(inputs)
                loss = F.cross_entropy(outputs, targets, reduction='none')
                losses_tensor[indexes] = loss.float().cpu()

        # Attach the losses tensor directly to the dataset object
        self.train_dset.irreducible_loss_cache = losses_tensor
        self.logger.info(f"Cached irreducible losses for {len(losses_tensor)} samples in dataset.")

    def test_teacher(self):
        self.logger.info('=====> Start teacher Validation')
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, datas in enumerate(self.test_loader):
                inputs = datas['input'].cuda()
                targets = datas['target'].cuda()
                outputs = self.teacher_model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        acc = np.mean(all_preds == all_labels)
        self.logger.info(f'=====> teacher Test Accuracy: {acc:.4f}')

    def get_ratio_per_epoch(self, epoch):
        if epoch < self.warmup_epochs:
            self.logger.info('warming up')
            return 1.0
        if self.ratio_scheduler == 'constant':
            return self.ratio
        elif self.ratio_scheduler == 'increase_linear':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == 'decrease_linear':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == 'increase_exp':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * np.exp(epoch / self.epochs)
        elif self.ratio_scheduler == 'decrease_exp':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * np.exp(epoch / self.epochs)
        else:
            raise NotImplementedError

    def reducible_loss_selection(self, inputs, targets, indexes, number_to_select, epoch):
        """Select sub-batch with highest reducible loss.
        Args:
            inputs (torch.Tensor): Input data for the current batch.
            targets (torch.Tensor): Corresponding target labels for the current batch.
        Returns:
            torch.Tensor: Indices of the selected samples.
        """
        # Set models to eval mode
        self.model.eval()

        # Get student loss from main model, irreducible loss from teacher model, and reducible loss by calculating the difference
        with torch.no_grad():
            total_loss = F.cross_entropy(self.model(inputs), targets, reduction='none')
        irreducible_loss = self.train_dset.irreducible_loss_cache[indexes].to(total_loss.device)
        reducible_loss = total_loss - irreducible_loss

        # Select samples with highest reducible loss
        _, index_selected = torch.topk(reducible_loss, k=number_to_select, largest=True, sorted=False)
        
        # Override with uniform selection if specified
        if epoch < self.uniform_epochs:
            self.logger.info('Uniform selection')
            index_selected = torch.randperm(len(inputs))[:number_to_select]
        
        # Return to train mode and return selected indices
        self.model.train()
        return index_selected.cpu().numpy(), reducible_loss.detach().cpu()

    def before_batch(self, i, inputs, targets, indexes, epoch):
        """Prepare the batch for training by selecting samples based on reducible loss.
        Args:
            i (int): Current batch index.
            inputs (torch.Tensor): Input data for the current batch.
            targets (torch.Tensor): Corresponding target labels for the current batch.
            indexes (torch.Tensor): Indices of the samples in the current batch.
            epoch (int): Current epoch number.
        Returns:
            tuple: Selected inputs, targets, and indexes for the current batch.
        """
        # Get the ratio for the current epoch
        ratio = self.get_ratio_per_epoch(epoch)
        if ratio == 1.0:
            if i == 0:
                self.logger.info('using all samples')
            return super().before_batch(i, inputs, targets, indexes, epoch)
        else:
            if i == 0:
                self.logger.info(f'balance: {self.balance}')
                self.logger.info('selecting samples for epoch {}, ratio {}'.format(epoch, ratio))

        # Get indices based on reducible loss
        number_to_select = max(1, int(inputs.shape[0] * ratio))
        indices, scores = self.reducible_loss_selection(inputs, targets, indexes, number_to_select, epoch)
        inputs = inputs[indices]
        targets = targets[indices]
        indexes = indexes[indices]
        return MinibatchInfo(inputs, targets, indexes, scores=scores)
