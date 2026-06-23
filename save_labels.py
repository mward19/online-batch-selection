import argparse
import os

import numpy as np
import torch

import data
from utils import get_configs


class _SimpleLogger: # must have to initialize dataset
    def info(self, msg):
        print(msg)


def _targets_to_numpy(targets):
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    else:
        targets = np.asarray(targets)

    if targets.ndim == 2:
        targets = targets.argmax(-1)

    return targets.astype(np.int32)


def _extract_targets(dataset, split_name):
    targets = getattr(dataset, "targets", None)
    if targets is None and hasattr(dataset, "dataset"):
        targets = getattr(dataset.dataset, "targets", None)

    if targets is None:
        raise AttributeError(f"{split_name} dataset does not expose a targets attribute")

    labels = _targets_to_numpy(targets)

    try:
        n_dataset = len(dataset)
        if labels.shape[0] != n_dataset:
            raise ValueError(
                f"{split_name} labels length {labels.shape[0]} does not match dataset length {n_dataset}"
            )
    except TypeError:
        pass

    return labels


def save_labels(train_dataset, test_dataset, out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    y_train = _extract_targets(train_dataset, "train")
    y_test = _extract_targets(test_dataset, "test")
    payload = {"train": y_train, "val": y_test}
    torch.save(payload, out_path)
    print(f"Saved labels: train={y_train.shape[0]}, test={y_test.shape[0]} -> {out_path}")


def build_config(args):
    data_config = get_configs(args.data)
    config = {**data_config}
    if "training_opt" not in config:
        config["training_opt"] = {}

    config["training_opt"].setdefault("batch_size", 1)
    config["training_opt"].setdefault("num_data_workers", 0)
    config["training_opt"].setdefault("test_batch_size", 1)
    return config


def main():
    parser = argparse.ArgumentParser(description="Save train/test labels once for a dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset config yaml")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name in output filename")
    parser.add_argument("--output", type=str, default=None, help="Optional explicit output path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite if output file exists")
    args = parser.parse_args()

    config = build_config(args)

    logger = _SimpleLogger() # must have to initialize dataset
    dataset_name = config["dataset"]["name"]
    dataset_fn = getattr(data, dataset_name)
    data_info = dataset_fn(config, logger)

    train_dataset = data_info["train_dset"]
    test_dataset = data_info.get("test_dset")
    if test_dataset is None and "test_loader" in data_info:
        test_dataset = data_info["test_loader"].dataset
    if test_dataset is None:
        raise KeyError("Could not find test dataset in data_info (expected test_dset or test_loader)")

    output_dataset_name = args.dataset if args.dataset is not None else dataset_name
    out_path = args.output if args.output is not None else f"labels/{output_dataset_name}.p"

    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists at {out_path}. Use --overwrite to replace it."
        )

    save_labels(train_dataset, test_dataset, out_path)


if __name__ == "__main__":
    main()