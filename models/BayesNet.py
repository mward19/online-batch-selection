import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import warnings
import numpy as np
import random
import clip
import math

import data.data_utils.mean_std as mean_std



class KFCALLAWrapper(nn.Module):
    def __init__(self, net, num_effective_data, prior_precision, n_f_samples, input_dim, last_layer_name="fc", momentum=0.99) :
        super(KFCALLAWrapper, self).__init__()
        self.net = net
        self.num_effective_data = num_effective_data
        self.prior_precision = prior_precision
        self.n_f_samples = n_f_samples
        self.momentum = momentum

        self.input_features_of_last_layer = None
        if isinstance(self.net, torch.nn.DataParallel):
            last_layer = getattr(self.net.module, last_layer_name)
        else:
            last_layer = getattr(self.net, last_layer_name)
        self.fhook = last_layer.register_forward_hook(self.forward_hook())
        # self.fhook = getattr(self.net, last_layer_name).register_forward_hook(self.forward_hook())

        with torch.no_grad():
            self.net.training = False
            out = self.net(torch.zeros(input_dim).cuda())
            self.net.training = True

            feature_dim = self.input_features_of_last_layer.shape[-1]
            out_dim = out.shape[-1]

        self.register_buffer("num_data", torch.Tensor([0]))
        self.register_buffer("A", torch.zeros(feature_dim, feature_dim))
        self.register_buffer("G", torch.zeros(out_dim, out_dim))
        self.register_buffer("G2", torch.zeros(out_dim, out_dim))
    
    def forward_hook(self):
        def hook(module, input, output):
            self.input_features_of_last_layer = input[0]
        return hook

    def forward(self, x, **kwargs):
        selection_pass = kwargs.get('selection_pass', False)
        y = kwargs.get('targets', None)

        bs = x.shape[0]
        if selection_pass:
            self.net.apply(_freeze)
        out = self.net(x)

        if selection_pass:
            self.net.apply(_unfreeze)

            if self.num_data.item() == 0:
                return out[:, None, :], out, None, None

            with torch.no_grad():
                V = math.sqrt(self.num_effective_data) * self.A
                V = V.clone()
                V.diagonal().add_(math.sqrt(self.prior_precision))
                L_V = psd_safe_cholesky(V)

                U = math.sqrt(self.num_effective_data) * self.G
                U = U.clone()
                U.diagonal().add_(math.sqrt(self.prior_precision))
                L_U = psd_safe_cholesky(U)

                V_inv = torch.cholesky_inverse(L_V)
                stds = (self.input_features_of_last_layer @ V_inv * self.input_features_of_last_layer).sum(-1).clamp(min=1e-6).sqrt()

                # Compute (L_U^T)^{-1} once
                out_dim = out.shape[-1]
                I = torch.eye(out_dim, device=out.device, dtype=out.dtype)
                L_U_T_inv = torch.linalg.solve_triangular(L_U.T, I, upper=True)

                L_f = stds.view(-1, 1, 1) * L_U_T_inv

                eps = torch.randn((bs, self.n_f_samples, out_dim), device=out.device, dtype=out.dtype)
                f_samples = out[:, None, :] + eps @ L_f

                # return f_samples, out, stds, torch.linalg.matrix_norm(L_U.T.inverse(), ord=2)  # for debugging, but computing SVD is expensive
                return f_samples, out, None, None
        elif self.training:
            assert y is not None, "Targets must be provided during training"
            with torch.no_grad():

                feature_cov = self.input_features_of_last_layer.T @ self.input_features_of_last_layer / bs
                if self.num_data.item() == 0:
                    self.A.data.copy_(feature_cov)
                else:
                    self.A.mul_(self.momentum).add_(feature_cov, alpha = 1-self.momentum)

                prob = out.softmax(-1)
                grad = prob - F.one_hot(y, out.shape[-1])
                grad_cov = grad.T @ grad / bs
                if self.num_data.item() == 0:
                    self.G.data.copy_(grad_cov)
                else:
                    self.G.mul_(self.momentum).add_(grad_cov, alpha = 1-self.momentum)
                
                # grad_cov2 = (prob.diag_embed() - prob[:, :, None] * prob[:, None, :]).mean(0)
                # if self.num_data.item() == 0:
                #     self.G2.data.copy_(grad_cov2)
                # else:
                #     self.G2.mul_(self.momentum).add_(grad_cov2, alpha = 1-self.momentum)
                self.num_data.add_(bs)
                # print(self.A[:10,:10], self.G[:10,:10], self.G2[:10,:10])

        return out

class CLIPZeroShotClassifier(nn.Module):
    def __init__(self, classnames, template, dataset, arch, tau):
        super(CLIPZeroShotClassifier, self).__init__()
        clip_model, preprocess = clip.load(arch, download_root='./models/teacher', jit=False)
        clip_model.eval()
        self.clip_model = clip_model
        clip_weights = clip_classifier(classnames, template, clip_model)
        self.register_buffer('clip_weights', clip_weights)            

        self.register_buffer('old_mean', torch.Tensor(mean_std.mean[dataset]))
        self.register_buffer('old_std', torch.Tensor(mean_std.std[dataset]))
        
        self.register_buffer('new_mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]))
        self.register_buffer('new_std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
        self.input_size = preprocess.transforms[0].size
        self.tau = tau
    
    @torch.no_grad()
    def forward(self, inputs):
        inputs = inputs.mul(self.old_std.view(-1, 1, 1)).add(self.old_mean.view(-1, 1, 1))
        if inputs.shape[1] == 1:
            # Convert grayscale to RGB
            inputs = inputs.repeat(1, 3, 1, 1)
        if inputs.shape[2] != self.input_size:
            inputs = F.interpolate(inputs, self.input_size, mode='bicubic')
        inputs = inputs.sub(self.new_mean.view(-1, 1, 1)).div(self.new_std.view(-1, 1, 1))

        input_features = self.clip_model.encode_image(inputs)
        clip_logits = self.tau * input_features @ self.clip_weights
        return clip_logits
    

def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def _freeze(m):
    if isinstance(m, (nn.BatchNorm2d)):
        m.track_running_stats = False

def _unfreeze(m):
    if isinstance(m, (nn.BatchNorm2d)):
        m.track_running_stats = True

def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        L = torch.linalg.cholesky(A, upper=upper, out=out)
        return L
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise ValueError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(10):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(
                    f"A not p.d., added jitter of {jitter_new} to the diagonal",
                    RuntimeWarning,
                )
                return L
            except RuntimeError:
                continue
        # return torch.randn_like(A).tril()
        raise e