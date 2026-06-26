import os
import math

import numpy as np
import torch
from scipy.stats import ortho_group
from sklearn.model_selection import train_test_split

import models

from .data_utils.generate_noise import apply_or_generate_label_noise


class wrapped_dataset(torch.utils.data.Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        if inputs.ndim != 2:
            raise ValueError(
                f"Teacher_Generated inputs must have shape (N, D), got {tuple(inputs.shape)}"
            )
        if targets.ndim != 1:
            raise ValueError(
                f"Teacher_Generated targets must have shape (N,), got {tuple(targets.shape)}"
            )
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Inputs and targets must have same length")

        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return {
            'input': self.inputs[index],
            'target': self.targets[index],
            'index': index,
        }


def _get_cache_path():
    return os.path.join(os.path.dirname(__file__), 'Teacher_Generated.p')


def _get_input_dim(dataset_cfg):
    input_dim = dataset_cfg.get('input_dim', dataset_cfg.get('in_channels'))
    if isinstance(input_dim, (list, tuple)):
        return math.prod(input_dim)
    return int(input_dim)


def _build_teacher_model(config):
    dataset_cfg = config.get('dataset', {})
    model_type = dataset_cfg['networks']['type']
    model_args = dict(dataset_cfg['networks'].get('params', {}))
    model_args['input_dim'] = dataset_cfg['input_dim']
    model_args['num_classes'] = 1
    return getattr(models, model_type)(**model_args)


def _load_cached_dataset(cache_path):
    payload = torch.load(cache_path, map_location='cpu')
    return (
        payload['X_train'].to(torch.float32),
        payload['y_train'].to(torch.long),
        payload['X_test'].to(torch.float32),
        payload['y_test'].to(torch.long),
    )


def _generate_and_save_dataset(config):
    dataset_cfg = config.get('dataset', {})
    seed = int(config.get('seed', 42))
    n_samples = int(dataset_cfg.get('n_samples', 10000))
    n_features = _get_input_dim(dataset_cfg)
    n_hidden = int(dataset_cfg.get('hidden_dim', 15))
    test_size = float(dataset_cfg.get('test_size', 0.2))
    teacher_model_path = dataset_cfg['generating_teacher_model_path']
    cache_path = _get_cache_path()

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    full_ortho = ortho_group.rvs(dim=n_features, random_state=rng)
    W = full_ortho[:n_hidden, :]

    a = np.zeros(n_hidden)
    strong = n_hidden // 2
    a[:strong] = rng.uniform(2.0, 3.0, size=strong)
    a[strong:] = rng.uniform(0.1, 0.5, size=n_hidden - strong)
    b1 = rng.uniform(-1.0, 1.0, size=n_hidden)

    Z = np.maximum(0.0, X @ W.T + b1)
    raw_logits = Z @ a
    scaling_factor = 1.0 / max(np.std(raw_logits), 1e-12)
    a_normalized = a * scaling_factor
    final_logits = Z @ a_normalized
    b2 = -np.mean(final_logits)
    probs = 1.0 / (1.0 + np.exp(-(final_logits + b2)))
    y = (probs > 0.5).astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32),
        y,
        test_size=test_size,
        random_state=seed,
    )

    teacher_model = _build_teacher_model(config)
    with torch.no_grad():
        teacher_model.linear1.weight.copy_(torch.from_numpy(W).float())
        teacher_model.linear1.bias.copy_(torch.from_numpy(b1).float())
        teacher_model.linear2.weight.copy_(torch.from_numpy(a_normalized).float().unsqueeze(0))
        teacher_model.linear2.bias.copy_(torch.tensor([b2], dtype=torch.float32))

    os.makedirs(os.path.dirname(teacher_model_path), exist_ok=True)
    torch.save(teacher_model.state_dict(), teacher_model_path)
    torch.save(
        {
            'X_train': torch.from_numpy(X_train),
            'y_train': torch.from_numpy(y_train),
            'X_test': torch.from_numpy(X_test),
            'y_test': torch.from_numpy(y_test),
        },
        cache_path,
    )

    return (
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )


def _load_or_generate_dataset(config, logger):
    dcfg = config.get('dataset', {})
    teacher_model_path = dcfg.get('generating_teacher_model_path', None)

    if teacher_model_path is None or str(teacher_model_path).strip() == '':
        raise ValueError(
            'dataset.generating_teacher_model_path must be provided for Teacher_Generated.'
        )

    cache_path = _get_cache_path()
    if os.path.isfile(teacher_model_path) and os.path.isfile(cache_path):
        X_train, y_train, X_test, y_test = _load_cached_dataset(cache_path)
        logger.info(f"Loaded cached Teacher_Generated dataset from {cache_path}")
    else:
        X_train, y_train, X_test, y_test = _generate_and_save_dataset(config)
        logger.info(f"Saved teacher model to {teacher_model_path}")
        logger.info(f"Saved Teacher_Generated dataset to {cache_path}")

    return X_train, y_train, X_test, y_test


def _build_test_loader(config, test_dset):
    config['training_opt']['test_batch_size'] = (
        config['training_opt']['batch_size']
        if 'test_batch_size' not in config['training_opt']
        else config['training_opt']['test_batch_size']
    )

    return torch.utils.data.DataLoader(
        test_dset,
        batch_size=config['training_opt']['test_batch_size'],
        shuffle=False,
        num_workers=config['training_opt']['num_data_workers'],
        pin_memory=True,
        drop_last=False,
    )


def Teacher_Generated(config, logger):
    """Synthetic dataset generated from a cached or newly created teacher model."""

    dcfg = config.get('dataset', {})
    n_samples = int(dcfg.get('n_samples', 10000))
    test_size = float(dcfg.get('test_size', 0.2))
    num_classes = 2

    if n_samples < 2:
        raise ValueError("Teacher_Generated requires n_samples >= 2")
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    X_train, y_train, X_test, y_test = _load_or_generate_dataset(config, logger)

    train_dset = wrapped_dataset(X_train, y_train)
    test_dset = wrapped_dataset(X_test, y_test)
    test_loader = _build_test_loader(config, test_dset)

    logger.info(
        "Teacher_Generated dataset created: "
        f"train={len(train_dset)}, test={len(test_dset)}, n_samples={n_samples}"
    )

    return {
        'num_classes': num_classes,
        'train_dset': train_dset,
        'train_dset_unaugmented': train_dset,
        'test_loader': test_loader,
        'num_train_samples': len(train_dset),
    }


def Teacher_Generated_Noise(config, logger):
    dcfg = config.get('dataset', {})
    n_samples = int(dcfg.get('n_samples', 10000))
    test_size = float(dcfg.get('test_size', 0.2))
    num_classes = 2

    if n_samples < 2:
        raise ValueError('Teacher_Generated requires n_samples >= 2')
    if not (0.0 < test_size < 1.0):
        raise ValueError(f'test_size must be in (0, 1), got {test_size}')

    X_train, y_train, X_test, y_test = _load_or_generate_dataset(config, logger)

    train_dset = wrapped_dataset(X_train, y_train)
    test_dset = wrapped_dataset(X_test, y_test)
    noise_metadata = apply_or_generate_label_noise(
        dataset=train_dset,
        num_classes=num_classes,
        dataset_config=config['dataset'],
        logger=logger,
        dataset_name='Teacher_Generated',
        seed=config.get('seed'),
        run_dir=config.get('save_dir'),
    )
    test_loader = _build_test_loader(config, test_dset)

    logger.info(
        'Teacher_Generated noisy dataset created: '
        f'train={len(train_dset)}, test={len(test_dset)}, n_samples={n_samples}'
    )

    return {
        'num_classes': num_classes,
        'train_dset': train_dset,
        'train_dset_unaugmented': train_dset,
        'test_loader': test_loader,
        'num_train_samples': len(train_dset),
        'noisy_indices': noise_metadata['noisy_indices'],
        'true_labels': noise_metadata['true_labels'],
    }