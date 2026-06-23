<h1 align="center">Online Batch Selection Methods for Training Acceleration</h1>

## Getting Started

### Install Project Dependencies

`Online-Batch-Selection` is managed via the `uv` package manager ([installation instructions](https://docs.astral.sh/uv/getting-started/installation/)). To install the dependencies, simply run `uv sync` from the root directory of the repository after cloning.

### Install Pre-Commit Hook

To install this repo's pre-commit hook with automatic linting and code quality checks, simply execute the following command:

```bash
pre-commit install
```

When you commit new code, the pre-commit hook will run a series of scripts to standardize formatting and run code quality checks. Any issues must be resolved for the commit to go through. If you need to bypass the linters for a specific commit, add the `--no-verify` flag to your git commit command.


## Data Preparation
For CIFAR datasets, the data will be automatically downloaded by the code.

For Tiny-ImageNet, please download the dataset from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and unzip it to the `_TINYIMAGENET` folder. Then, run the following command to prepare the data:
```bash
cd _TINYIMAGENET
python val_folder.py
```

## Running
```bash
CUDA_VISIBLE_DEVICES=0 uv run main.py \
  --method configs/cifar10/method/uniform-0.1.yaml \
  --data configs/cifar10/data/cifar10.yaml \
  --model configs/cifar10/model/resnet18.yaml \
  --optim configs/cifar10/optim/adamw-320-0.001-0.01.yaml \
  --diagnostics configs/cifar10/diagnostics/all_log_interval.yaml \
  --wandb_not_upload
```
The `--wandb_not_upload` flag is optional and keeps wandb logs local instead of uploading them.

`CUDA_VISIBLE_DEVICES` selects visible GPU devices (for example `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES="0,2"`).

## Save Labels Once

You can generate and cache train/val labels once per dataset using [save_labels.py](save_labels.py). This writes:

- `results/data/labels_{dataset}.p`

General form:

```bash
uv run save_labels.py --data <dataset-config-yaml>
```

Dataset examples:

```bash
uv run save_labels.py --data configs/mnist/data/mnist.yaml
uv run save_labels.py --data configs/fashionmnist/data/fashionmnist.yaml
uv run save_labels.py --data configs/cifar10/data/cifar10.yaml
uv run save_labels.py --data configs/cifar100/data/cifar100.yaml
uv run save_labels.py --data configs/cifar3/data/cifar3.yaml
uv run save_labels.py --data configs/tinyimagenet/data/tinyimagenet.yaml
uv run save_labels.py --data configs/twomoons/data/twomoons.yaml
uv run save_labels.py --data configs/makeblobs/data/makeblobs.yaml
```

Useful flags:

- `--overwrite`: replace an existing labels file.
- `--output <path>`: write to a custom path.
- `--batch_size <int>` and `--num_workers <int>`: control export loader settings.

## Repository Structure

The repository is organized as follows:

### Core Components

- **`methods/SelectionMethod.py`** - Parent class containing the main training loop and core batch selection logic. All selection methods inherit from this base class and implement `before_batch` behavior.

- **`methods/`** - Implementations of batch selection methods. Current methods include Full, Uniform, DivBS, RhoLoss, Bayesian, TrainLoss, GradNorm, GradNormIS, and Optk.

- **`methods/method_utils/`** - Shared training utilities (diagnostics, NTK logging, optimizer/scheduler helpers, probes, snapshots).

- **`configs/`** - Configuration tree grouped by dataset (`cifar10`, `cifar100`, `mnist`, etc.) with separate `data`, `method`, `model`, `optim`, and `diagnostics` YAMLs.

  Example method config snippet:

  ```yaml
  method: Uniform
  method_opt:
    ratio: 0.1
  ```

  > **Note:** `RhoLoss` and `Bayesian` require additional hyperparameters.

- **`main.py`** - Entry point for running experiments. Handles argument parsing and experiment initialization.

### Data

- **`_TINYIMAGENET/`** - Tiny-ImageNet dataset directory (see [Data Preparation](#data-preparation)).
- **`_CIFAR/`** - CIFAR datasets (automatically downloaded).

### Experiments

- **`exp/`** - Experiment results and logs (git-ignored).
- **`exports/`** - Exported analysis artifacts for completed runs.
- **`results/`** - Aggregated outputs used by plotting/post-processing scripts.
- **`wandb/`** - Weights & Biases logging directory (git-ignored).

### Key Files to Review

1. Start with `methods/SelectionMethod.py` to understand the training loop architecture
2. Explore `methods/` to see specific batch selection implementations
3. Check `configs/` for experiment configuration options
4. Review `methods/method_utils/diagnostics.py` and `methods/method_utils/ntk.py` for diagnostics/NTK behavior

## Development

### Managing Dependencies

To add a new dependency to the project, run `uv add <package-name>`. This will install the dependency into uv's managed .venv and automatically update the `pyproject.toml` file and the `uv.lock` file, ensuring that the dependency is available for all users of the project who run `uv sync`.

To remove a dependency, run `uv remove <package-name>`. This will perform the reverse of `uv add` (including updating the `pyproject.toml` and `uv.lock` files).

See [uv's documentation](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) for more details.

## Optional: Enable TRAK / TRAKer for Projection NTK Diagnostics

Projection NTK variants (`proj-pseudo`, `proj-trace`) require `TRAKer` from `trak`, which may not be present in the default project dependency set.

Install it in your current `uv` environment:

```bash
uv pip install traker
```

If you specifically need the PNNL projection NTK implementation, install it from source:

```bash
uv pip install "git+https://github.com/pnnl/projection_ntk.git"
```

Quick verification:

```bash
uv run python -c "from trak import TRAKer; print('TRAKer import OK')"
```

## TODO

- [ ] Fix multiple GPU parallelization.
- [ ] Fix saving/loading training state so runs can be resumed reliably (including diagnostics state in `methods/method_utils/diagnostics.py`).
- [ ] Fix run naming to avoid output overwrites.
- [ ] Merge in Connor's code.
