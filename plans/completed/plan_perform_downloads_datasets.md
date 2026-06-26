# Plan: download datasets in `perform_downloads.py`

~~Implemented (TinyImageNet auto-download skipped per user).~~

## Problem

`perform_downloads.py` runs on the login node (which has network) to pre-fetch
anything a compute node can't reach. Today it only fetches the CLIP teacher.
Datasets are downloaded lazily inside the data loaders via torchvision's
`download=True`, which fails on a compute node with `Network is unreachable`
(the MNIST traceback the user hit). We need `perform_downloads.py` to also
pre-fetch the dataset named in the config.

## How the loaders download (survey)

Every torchvision-backed loader calls `datasets.<Class>(config['dataset']['root'],
train=..., download=True, ...)` for both train and test. Grouping the config
dataset names (which match loader function names in `data/__init__.py`) by the
torchvision class they hit:

| torchvision class | config dataset names |
|---|---|
| `MNIST` | MNIST, MNIST_Noise, MNIST10, MNIST90, MNIST90_Noise |
| `FashionMNIST` | FashionMNIST, FashionMNIST_Noise |
| `CIFAR10` | CIFAR3, CIFAR3_Noise, CIFAR10, CIFAR10_minimal, CIFAR10_Noise, CIFAR10_LT |
| `CIFAR100` | CIFAR100, CIFAR100_Noise, CIFAR100_LT |

Two other groups need no network fetch here:

- **Synthetic** — `MakeBlobs`, `MakeBlobs_Noise`, `Teacher_Generated`,
  `Teacher_Generated_Noise`: generated in-process, nothing to download.
- **TinyImageNet** (`TinyImageNet`, `TinyImageNet_Noise`): the loader's own
  download is commented out and it `raise`s "Dataset not found"
  (`data/tinyimagenet.py:62-82`). It expects a pre-extracted
  `tiny-imagenet-200/` under `root`. Out of scope for this change — see note.

Downloading is idempotent: torchvision checks the existing files' integrity and
skips the fetch if they're already present, so re-running is safe.

## Changes (all in `perform_downloads.py`)

### 1. Name → torchvision class map

Add a module-level dict mapping each config dataset name to its torchvision
class name:

```python
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
```

### 2. `download_dataset(config_path)` function

```python
def download_dataset(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    try:
        name = config["dataset"]["name"]
        root = config["dataset"]["root"]
    except (KeyError, TypeError):
        print("No dataset name/root in config. Not downloading dataset.")
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
```

Notes:
- No `transform` is passed — we only need the bytes on disk; the loader applies
  transforms later.
- `torchvision` is imported inside the function, matching the lazy `import clip`
  style already used by `download_clip`.

### 3. Call it from `__main__`

Keep the existing `--method` arg (it's the merged config path that
`download_clip` already reads) and call both:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    download_clip(args.method)
    download_dataset(args.method)
```

(The submit scripts already invoke
`python perform_downloads.py --method <config>` per config, e.g.
`run_mnist.py:34`, so no caller change is needed.)

## Out of scope

- TinyImageNet auto-download (its in-loader download is intentionally disabled).
  {{Want me to also wire up the TinyImageNet zip download
  (`http://cs231n.stanford.edu/tiny-imagenet-200.zip`) here? It's a ~248MB
  fetch + unzip, separate from the torchvision path.}}
- Renaming the `--method` flag (it's really a config path now), to avoid
  touching the submit scripts in this change.

## Manual verification

On a login node (network available):

```bash
python perform_downloads.py --method configs/mnist_basic.yaml   # or a configs-temp/ file
```

Confirm MNIST files appear under the config's `dataset.root`, then re-run to
confirm it skips (idempotent).
