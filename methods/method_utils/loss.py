import torch
import torch.nn as nn
import torch.nn.functional as F 

def wrap_criterion(criterion):
    """
    Returns a wrapped version of criterion with a reduction parameter, so that
    the criterion doesn't have to be reinstantiated to change the reduction
    type.
    """
    def wrapped_criterion(input, target, *args, reduction='mean', weights=None, **kwargs):
        losses = criterion(input, target, *args, **kwargs)

        if reduction == 'mean':    
            return torch.mean(losses)
        
        elif reduction == 'weighted':
            if weights is None:
                raise ValueError(f"To use weighted reduction, you must pass weights of the same length as the input and target")
            return torch.sum(weights * losses)
        
        elif reduction == 'none':
            return losses
        
        else:
            raise ValueError(f"Reduction {reduction} not implemented")
        
    return wrapped_criterion


def create_criterion(config, logger):
    loss_type = config['training_opt']['loss_type']
    loss_params = config['training_opt']['loss_params']
    if loss_type == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(reduction='none', **loss_params)
    elif loss_type == "FocalLoss":
        if 'gamma' not in loss_params:
            raise ValueError(f"FocalLoss requires a 'gamma' parameter in loss_params, but got {loss_params}")
        def focal_loss(logits, targets, gamma=loss_params['gamma']):
            log_probs = F.log_softmax(logits, dim=-1)

            log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt = log_pt.exp()

            loss = -((1 - pt) ** gamma) * log_pt

            return loss
        criterion = focal_loss
    elif loss_type == "Squentropy":
        def squentropy_loss(logits, targets):
            probs = F.softmax(logits, dim=-1)  # [B, C]

            targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

            loss = (probs - targets_one_hot) ** 2
            loss = loss.sum(dim=-1)  # per-sample scalar [B]

            return loss
        criterion = squentropy_loss
    else:
        raise NotImplementedError
    
    return wrap_criterion(criterion)

def create_teacher_criterion(config, logger):
    teacher_loss_type = config['rholoss']['teacher_loss_type']
    teacher_loss_params = config['rholoss']['teacher_loss_params']
    if teacher_loss_type == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(reduction='none', **teacher_loss_params)
    else:
        raise NotImplementedError

    return wrap_criterion(criterion)