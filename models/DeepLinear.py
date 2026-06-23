import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def create_model(m_type='linear', input_dim=[1,1,10], num_classes=10, pretrained=False, *, activation=nn.Identity(), **kwargs):
    input_dim_scalar = math.prod(input_dim)

    if 'hidden_dims' in kwargs and ('num_hidden_layers' in kwargs or 'hidden_dim' in kwargs):
        raise ValueError(
            "Cannot specify exact hidden dimensions (hidden_dims) and "
            "number of hidden layers (num_hidden_layers) or hidden dimension "
            "to repeat (hidden_dim) at the same time"
        )
    
    if 'num_hidden_layers' in kwargs:
        if 'hidden_dim' not in kwargs:
            raise ValueError('Must specify hidden dimension if number of hidden layers is specified')
        hidden_dims = [kwargs['hidden_dim']] * kwargs['num_hidden_layers']
    elif 'hidden_dims' in kwargs:
        hidden_dims = kwargs['hidden_dims']
    else:
        raise ValueError('You must specify hidden dimension information')

    model = DeepLinear(
        input_dim=input_dim_scalar, 
        hidden_dims=hidden_dims, 
        num_classes=num_classes,
        activation=activation
    )
    return model

def create_model_relu(*args, **kwargs):
    return create_model(*args, activation=nn.ReLU(), **kwargs)

class DeepLinear(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, flatten_input=True, activation=nn.Identity()):
        super().__init__()

        all_dims = [input_dim] + hidden_dims + [num_classes]

        self.hidden = nn.Sequential(*sum(
            (
                [nn.Linear(d1, d2), activation] 
                for d1, d2 in zip(all_dims[:-2], all_dims[1:-1])
            ),
            start=[]
        ))

        self.classifier = nn.Linear(all_dims[-2], all_dims[-1])

        # Orthogonal init avoids vanishing forward/backward signal that default init causes at large depth
        gain = nn.init.calculate_gain('relu') if isinstance(activation, nn.ReLU) else 1.0
        for layer in self.hidden:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.orthogonal_(self.classifier.weight)

        self.flatten_input = flatten_input

    def forward(self, x, *, last_layer_grad_only=False, **kwargs):
        with torch.set_grad_enabled(not last_layer_grad_only):
            if self.flatten_input:
                x = x.flatten(1)

            x = self.hidden(x)
        
        feat = x
        x = self.classifier(x)

        if kwargs.get("need_features", False):
            return x, feat
        return x
    
    def feat_nograd_forward(self, x, **kwargs):
        return self.forward(x, last_layer_grad_only=True, need_features=True, **kwargs)