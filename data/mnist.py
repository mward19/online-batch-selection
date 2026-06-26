from torchvision import datasets, transforms
import os
import torch
import numpy as np
import random

from .data_utils.generate_noise import apply_or_generate_label_noise

mnist_classes = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]

mnist_templates = [
    "a photo of the number {}.",
    "a handwritten digit {}.",
    "a grayscale image of the number {}.",
    "a photo of a handwritten {}.",
]

fashionmnist_classes = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

fashionmnist_templates = [
    "a photo of a {}.",
    "a grayscale photo of a {}.",
    "a photo of a person wearing a {}.",
    "a photo of a {} on a white background.",
    "a photo of a {} item.",
]

class wrapped_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # do not cache targets here; access dataset.targets at call-time to reflect updates
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        sample = self.dataset[index]
        inp = sample[0]
        # Prefer explicit targets attribute (works for Subset where we set subset.targets),
        # otherwise fall back to the label returned by the underlying dataset.__getitem__.
        if hasattr(self.dataset, 'targets'):
            tgt = self.dataset.targets[index]
        else:
            tgt = sample[1]
        # coerce tensor->int for consistency
        if isinstance(tgt, torch.Tensor):
            try:
                tgt = int(tgt.item())
            except Exception:
                tgt = int(tgt)
        return {
            'input': inp,
            'target': tgt,
            'index': index
        }


def _build_test_loader(config, dst_test):
    config['training_opt']['test_batch_size'] = config['training_opt']['batch_size'] if 'test_batch_size' not in config['training_opt'] else config['training_opt']['test_batch_size']
    return torch.utils.data.DataLoader(
        wrapped_dataset(dst_test), batch_size = config['training_opt']['test_batch_size'],
        shuffle=False, num_workers = config['training_opt']['num_data_workers'], pin_memory=True, drop_last=False
    )


def _build_dataset_info(config, logger, dataset_name, dst_train, dst_test, num_classes, classes, templates, include_noise=False):
    payload = {
        'num_classes': num_classes,
        'train_dset': wrapped_dataset(dst_train),
        'test_loader': _build_test_loader(config, dst_test),
        'num_train_samples': len(dst_train),
        'classes': classes,
        'template': templates,
    }
    if include_noise:
        payload.update(
            apply_or_generate_label_noise(
                dataset=dst_train,
                num_classes=num_classes,
                dataset_config=config['dataset'],
                logger=logger,
                dataset_name=dataset_name,
                seed=config.get('seed'),
                run_dir=config.get('save_dir'),
            )
        )
        payload['train_dset'] = wrapped_dataset(dst_train)
    return payload
    
def MNIST(config, logger):
    im_size = (28, 28) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 28 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 28 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )
    
    dst_train = datasets.MNIST(
        config['dataset']['root'], train=True, download=True, transform= transform
    )
    
    
    dst_test = datasets.MNIST(config['dataset']['root'], train=False, download=True, transform=test_transform)
    # class_names = dst_train.classes
    # dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
    # dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)
    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='MNIST',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=mnist_classes,
        templates=mnist_templates,
    )


def MNIST_Noise(config, logger):
    im_size = (28, 28) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 28 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 28 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    dst_train = datasets.MNIST(
        config['dataset']['root'], train=True, download=True, transform= transform
    )

    dst_test = datasets.MNIST(config['dataset']['root'], train=False, download=True, transform=test_transform)

    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='MNIST',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=mnist_classes,
        templates=mnist_templates,
        include_noise=True,
    )


def MNIST10(config, logger):
    payload = MNIST(config, logger)
    if 'subset_idx_path' not in config['dataset']:
        raise ValueError('config["dataset"]["subset_idx_path"] must be set to .npy file of 10% indices')
    idx_file = config['dataset']['subset_idx_path']
    indices_10 = np.load(idx_file)
    indices_10 = np.array(indices_10, dtype=np.int64)

    orig_wrapped = payload['train_dset']
    orig_dataset = orig_wrapped.dataset
    # create subset for the specified 10%
    subset = torch.utils.data.Subset(orig_dataset, indices_10.tolist())
    try:
        subset.targets = torch.tensor(np.array(orig_dataset.targets)[indices_10], dtype=torch.long)
    except Exception:
        subset.targets = None

    payload['train_dset'] = wrapped_dataset(subset)
    payload['num_train_samples'] = len(subset)
    return payload


def MNIST90(config, logger):
    payload = MNIST(config, logger)
    if 'subset_idx_path' not in config['dataset']:
        raise ValueError('config["dataset"]["subset_idx_path"] must be set to .npy file of 10% indices')
    idx_file = config['dataset']['subset_idx_path']
    indices_10 = np.load(idx_file)
    indices_10 = np.array(indices_10, dtype=np.int64)

    orig_wrapped = payload['train_dset']
    orig_dataset = orig_wrapped.dataset
    all_idx = np.arange(len(orig_dataset))
    indices_90 = np.setdiff1d(all_idx, indices_10)

    subset = torch.utils.data.Subset(orig_dataset, indices_90.tolist())
    try:
        subset.targets = torch.tensor(np.array(orig_dataset.targets)[indices_90], dtype=torch.long)
    except Exception:
        subset.targets = None

    payload['train_dset'] = wrapped_dataset(subset)
    payload['num_train_samples'] = len(subset)
    return payload


def MNIST90_Noise(config, logger):
    # Build base MNIST payload and create 90% subset first (mirror MNIST_Noise flow)
    payload = MNIST(config, logger)
    if 'subset_idx_path' not in config['dataset']:
        raise ValueError('config["dataset"]["subset_idx_path"] must be set to .npy file of 10% indices')
    idx_file = config['dataset']['subset_idx_path']
    indices_10 = np.load(idx_file)
    indices_10 = np.array(indices_10, dtype=np.int64)

    orig_wrapped = payload['train_dset']
    orig_dataset = orig_wrapped.dataset
    all_idx = np.arange(len(orig_dataset))
    indices_90 = np.setdiff1d(all_idx, indices_10)

    subset = torch.utils.data.Subset(orig_dataset, indices_90.tolist())
    try:
        subset.targets = torch.tensor(np.array(orig_dataset.targets)[indices_90], dtype=torch.long)
    except Exception:
        subset.targets = None

    # apply noise to the subset (after selection), mirroring MNIST_Noise's include_noise behavior.
    # The cache name is keyed by dataset_name ('MNIST90_Noise'), so the subset gets its own
    # cache file distinct from the full dataset -- no stale-collision removal needed.
    noise_meta = apply_or_generate_label_noise(
        dataset=subset,
        num_classes=10,
        dataset_config=config['dataset'],
        logger=logger,
        dataset_name='MNIST90_Noise',
        seed=config.get('seed'),
        run_dir=config.get('save_dir'),
    )

    payload['train_dset'] = wrapped_dataset(subset)
    payload['num_train_samples'] = len(subset)

    # apply_or_generate_label_noise returns true_labels and noisy_indices (as tensors)
    if isinstance(noise_meta, dict):
        payload.update(noise_meta)

    return payload

def FashionMNIST(config, logger):
    im_size = (28, 28) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    mean = [0.2860]
    std = [0.3530]

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 28 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 28 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )
    
    dst_train = datasets.FashionMNIST(
        config['dataset']['root'], train=True, download=True, transform= transform
    )
    
    
    dst_test = datasets.FashionMNIST(config['dataset']['root'], train=False, download=True, transform=test_transform)
    # class_names = dst_train.classes
    # dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
    # dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)
    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='FashionMNIST',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=fashionmnist_classes,
        templates=fashionmnist_templates,
    )


def FashionMNIST_Noise(config, logger):
    im_size = (28, 28) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    mean = [0.2860]
    std = [0.3530]

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 28 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 28 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    dst_train = datasets.FashionMNIST(
        config['dataset']['root'], train=True, download=True, transform= transform
    )

    dst_test = datasets.FashionMNIST(config['dataset']['root'], train=False, download=True, transform=test_transform)

    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='FashionMNIST',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=fashionmnist_classes,
        templates=fashionmnist_templates,
        include_noise=True,
    )