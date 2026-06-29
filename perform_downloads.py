"""
To be used before a run on a supercomputer compute node, to download necessary
data on a login node. Right now, just downloads the CLIP model
"""


import os
import yaml
import argparse

# Config dataset name -> torchvision.datasets class. These loaders fetch via
# download=True at load time; we pre-fetch the same files here on the login node.
_TORCHVISION_DATASETS = {
    "MNIST": "MNIST", "MNIST_Noise": "MNIST", "MNIST10": "MNIST",
    "MNIST90": "MNIST", "MNIST90_Noise": "MNIST",
    "FashionMNIST": "FashionMNIST", "FashionMNIST_Noise": "FashionMNIST",
    "CIFAR3": "CIFAR10", "CIFAR3_Noise": "CIFAR10", "CIFAR10": "CIFAR10",
    "CIFAR10_minimal": "CIFAR10", "CIFAR10_Noise": "CIFAR10",
    "CIFAR10_LT": "CIFAR10",
    "CIFAR100": "CIFAR100", "CIFAR100_Noise": "CIFAR100",
    "CIFAR100_LT": "CIFAR100",
}

# Synthetic datasets generated in-process — nothing to pre-fetch.
_SYNTHETIC_DATASETS = {
    "MakeBlobs", "MakeBlobs_Noise", "Teacher_Generated", "Teacher_Generated_Noise",
}

def download_clip(method_config_path):
    with open(method_config_path) as f:
        config = yaml.safe_load(f)
    try:
        arch = config["clip"]["clip_architecture"]
    except (KeyError, TypeError):
        print(f"CLIP architecture not listed in config. Not downloading.")
        return

    filename = arch.replace("/", "-") + ".pt"
    path = os.path.join("./models/teacher", filename)
    if os.path.exists(path):
        print(f"CLIP arch {arch} already present at {path}. Skipping download.")
        return

    print(f"Downloading CLIP arch: {arch}")
    import clip
    clip.load(arch, download_root="./models/teacher", jit=False, device="cpu")
    print("Done.")

def download_dataset(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    try:
        name = config["dataset"]["name"]
        root = config["dataset"]["root"]
    except (KeyError, TypeError):
        print("No dataset name/root in config. Not downloading dataset.")
        return
    
    if os.path.exists(root):
        print(f"Directory {root} already exists, not downloading dataset {name}.")
        return

    if name in _SYNTHETIC_DATASETS:
        print(f"Dataset {name} is generated in-process. Nothing to download.")
        return

    cls_name = _TORCHVISION_DATASETS.get(name)
    if cls_name is None:
        print(f"Dataset {name} has no pre-fetch rule (e.g. TinyImageNet must be "
              f"staged manually). Skipping.")
        return

    from torchvision import datasets
    cls = getattr(datasets, cls_name)
    print(f"Downloading dataset {name} ({cls_name}) into {root}")
    cls(root, train=True, download=True)
    cls(root, train=False, download=True)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    download_clip(args.method)
    download_dataset(args.method)