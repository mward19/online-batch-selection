import numpy as np
import torch

from .data_utils.generate_noise import apply_or_generate_label_noise


makeblobs_templates = [
    'a point from class {}.',
]


class wrapped_dataset(torch.utils.data.Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        if inputs.ndim != 2:
            raise ValueError(f"MakeBlobs inputs must have shape (N, D), got {tuple(inputs.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"MakeBlobs targets must have shape (N,), got {tuple(targets.shape)}")
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


def _makeblobs_output(config, logger, train_dset, test_dset, num_classes, include_noise=False, wstar_test_acc=None, what_test_acc=None):
    payload = {
        'num_classes': num_classes,
        'train_dset': train_dset,
        'train_dset_unaugmented': train_dset,
        'test_loader': torch.utils.data.DataLoader(
            test_dset,
            batch_size=config['training_opt']['test_batch_size'],
            shuffle=False,
            num_workers=config['training_opt']['num_data_workers'],
            pin_memory=True,
            drop_last=False,
        ),
        'num_train_samples': len(train_dset),
        'classes': [f'class {class_idx}' for class_idx in range(num_classes)],
        'template': makeblobs_templates,
    }
    if include_noise:
        payload.update(
            apply_or_generate_label_noise(
                dataset=train_dset,
                num_classes=num_classes,
                dataset_config=config['dataset'],
                logger=logger,
                dataset_name='MakeBlobs',
                seed=config.get('seed'),
                run_dir=config.get('save_dir'),
            )
        )
    if wstar_test_acc is not None:
        payload['wstar_test_acc'] = wstar_test_acc
    if what_test_acc is not None:
        payload['what_test_acc'] = what_test_acc
    return payload


def MakeBlobs(config, logger):
    """Synthetic Gaussian blobs dataset.

    Returns the same dict structure as other loaders:
    - train_dset: torch Dataset yielding {'input','target','index'}
    - test_loader: DataLoader over test set

    Config options (dataset.*):
    - n_samples (int): total samples (default 10000)
    - n_features (int): feature dimension (default 2)
    - centers (int | list[list[float]]): number or coordinates of centers (default 3)
    - cluster_std (float | list[float]): std dev for each cluster (default 1.0)
    - center_box (list[float,float]): bounds for random centers (default [-10.0, 10.0])
    - test_size (float): fraction for test split (default 0.2)
    - standardize (bool): z-score features using train stats (default True)
    - random_state (int|None): overrides config['seed'] if set
    """

    try:
        from sklearn.datasets import make_blobs
        from sklearn.model_selection import train_test_split
    except Exception as e:
        raise ImportError(
            "MakeBlobs requires scikit-learn. Install it (e.g., `pip install scikit-learn`)."
        ) from e

    dcfg = config.get('dataset', {})
    seed = dcfg.get('random_state', config.get('seed', 16))

    n_samples = int(dcfg.get('n_samples', 10_000))
    n_features = int(dcfg.get('n_features', 2))
    centers = dcfg.get('centers', 3)
    centers_type = dcfg.get('centers_type', None)
    if centers_type == 'from_file':
        center_file = dcfg['center_file']
        centers = np.load(center_file)  # shape (2, n_features)
        if centers.shape != (2, n_features):
            raise ValueError(f'center_file has shape {centers.shape}, expected (2, {n_features})')
        num_classes = 2
    elif centers_type is not None:
        raise ValueError(f'Unknown centers_type: {centers_type}')
    elif isinstance(centers, int):
        num_classes = centers
    else:
        num_classes = len(centers)
    cluster_std = dcfg.get('cluster_std', 1.0)
    center_box = dcfg.get('center_box', [-10.0, 10.0])
    test_size = float(dcfg.get('test_size', 0.2))
    standardize = bool(dcfg.get('standardize', True))

    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        center_box=tuple(center_box),
        random_state=seed,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Evaluate w_star and w_hat classifiers on raw X_test before standardization
    # so no coordinate transformation is needed (both vectors live in raw feature space).
    wstar_test_acc = None
    wstar_file = dcfg.get('wstar_file')
    if wstar_file is not None:
        w_star = np.load(wstar_file).astype(np.float32)
        z = X_test @ w_star
        preds = (z <= 0).astype(np.int64)  # z>0 → near +w_star → label 0
        wstar_test_acc = float((preds == y_test).mean())

    what_test_acc = None
    wnoised_file = dcfg.get('wnoised_file')
    if wnoised_file is not None:
        w_hat = np.load(wnoised_file).astype(np.float32)
        z = X_test @ w_hat
        preds = (z <= 0).astype(np.int64)  # same sign convention as w_star
        what_test_acc = float((preds == y_test).mean())

    if standardize:
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-6)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    train_dset = wrapped_dataset(X_train_t, y_train_t)
    test_dset = wrapped_dataset(X_test_t, y_test_t)

    config['training_opt']['test_batch_size'] = (
        config['training_opt']['batch_size']
        if 'test_batch_size' not in config['training_opt']
        else config['training_opt']['test_batch_size']
    )

    return _makeblobs_output(
        config=config,
        logger=logger,
        train_dset=train_dset,
        test_dset=test_dset,
        num_classes=num_classes,
        wstar_test_acc=wstar_test_acc,
        what_test_acc=what_test_acc,
    )


def MakeBlobs_Noise(config, logger):
    try:
        from sklearn.datasets import make_blobs
        from sklearn.model_selection import train_test_split
    except Exception as e:
        raise ImportError(
            "MakeBlobs requires scikit-learn. Install it (e.g., `pip install scikit-learn`)."
        ) from e

    dcfg = config.get('dataset', {})
    seed = dcfg.get('random_state', config.get('seed', 16))

    n_samples = int(dcfg.get('n_samples', 10_000))
    n_features = int(dcfg.get('n_features', 2))
    centers = dcfg.get('centers', 3)
    centers_type = dcfg.get('centers_type', None)
    if centers_type == 'from_file':
        center_file = dcfg['center_file']
        centers = np.load(center_file)  # shape (2, n_features)
        if centers.shape != (2, n_features):
            raise ValueError(f'center_file has shape {centers.shape}, expected (2, {n_features})')
        num_classes = 2
    elif centers_type is not None:
        raise ValueError(f'Unknown centers_type: {centers_type}')
    elif isinstance(centers, int):
        num_classes = centers
    else:
        num_classes = len(centers)
    cluster_std = dcfg.get('cluster_std', 1.0)
    center_box = dcfg.get('center_box', [-10.0, 10.0])
    test_size = float(dcfg.get('test_size', 0.2))
    standardize = bool(dcfg.get('standardize', True))

    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        center_box=tuple(center_box),
        random_state=seed,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    if standardize:
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-6)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    train_dset = wrapped_dataset(X_train_t, y_train_t)
    test_dset = wrapped_dataset(X_test_t, y_test_t)

    config['training_opt']['test_batch_size'] = (
        config['training_opt']['batch_size']
        if 'test_batch_size' not in config['training_opt']
        else config['training_opt']['test_batch_size']
    )

    return _makeblobs_output(
        config=config,
        logger=logger,
        train_dset=train_dset,
        test_dset=test_dset,
        num_classes=num_classes,
        include_noise=True,
    )
