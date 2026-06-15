# Plan: Switch Bayesian config to hyperplane teacher

## Goal

Update `configs/makeblobs/method/bayesian-0.1.yaml` to use the same local pretrained hyperplane teacher as `configs/makeblobs/method/rholoss-0.1-hyperplane.yaml`, instead of CLIP.

## ~~Change~~

In `configs/makeblobs/method/bayesian-0.1.yaml`, replace:

```yaml
teacher_model_source: clip # Supported options: "local_pretrained", "clip", "timm"
teacher_model_path: /home/lgreen/projects/Online_BS/models/teacher/MakeBlobs.tar
# teacher_model_path: /home/phancock/Online-Batch-Selection/exp/cifar10_small-cnn_30-epoch.tar

# If local_pretrained is selected, specify the model architecture here
local_pretrained:
  type: Small_cnn
  params: 
    m_type: 'Small_cnn'

# If Clip is selected, specify the model architecture here
clip:
  clip_architecture: 'RN50'
  tau: 4
```

with:

```yaml
teacher_model_source: local_pretrained
teacher_model_path: models/teacher/makeblobs_1024d_hyperplane_alpha1.0_nseed0.pth

local_pretrained:
  type: Linear
  params:
    m_type: linear
```

All other fields (`method`, `method_opt`, Online Laplace Approximation Parameters) remain unchanged.
