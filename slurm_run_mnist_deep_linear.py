from itertools import product
from pathlib import Path
from textwrap import dedent
from datetime import datetime
from tqdm import tqdm
import shlex
import subprocess
import re

WANDB_PROJECT = "Matthew—Deep Linear Networks (Blobs)"

USE_SLURM = True
EXP_BASE = "./exp/"  # change to e.g. "./exp-ablation/" to redirect output

# SEEDS = [1, 2, 3]
SEEDS = [1]
DIAGNOSTICS = "configs/diagnostics/snapshots_log_interval.yaml"
CONFIG_DIR = "configs/mnist"

METHODS = [
    # f"{CONFIG_DIR}/method/rholoss-0.1.yaml",
    # f"{CONFIG_DIR}/method/bayesian-0.1.yaml",
    # f"{CONFIG_DIR}/method/divbs-0.1.yaml",
    f"{CONFIG_DIR}/method/uniform-0.1.yaml",
]

MODELS = [
    f"{CONFIG_DIR}/model/deep_linear_saxe/deep_linear_{i}.yaml"
    # for i in [3, 5, 8, 36, 100]
    for i in [3]
]

OPTIMS = [f"{CONFIG_DIR}/optim/adamw-320-0.001-0.01.yaml"]
DATAS = [f"{CONFIG_DIR}/data/mnist.yaml"]

Path("logs").mkdir(exist_ok=True)

save_dirs_file = (
    Path("logs")
    / f"save_dirs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

jobs = list(product(SEEDS, DATAS, MODELS, OPTIMS, METHODS))

with open(save_dirs_file, "w") as f:
    for seed, data, model, optim, method in tqdm(
        jobs,
        desc="Submitting jobs" if USE_SLURM else "Running jobs",
        total=len(jobs),
    ):
        subprocess.run(
            [
                "python",
                "perform_downloads.py",
                "--method",
                method,
            ],
            check=True,
        )

        save_dir = subprocess.check_output(
            [
                "python",
                "get_save_dir.py",
                "--method", method,
                "--data", data,
                "--model", model,
                "--optim", optim,
                "--seed", str(seed),
                "--exp_base", EXP_BASE,
            ],
            text=True,
        ).strip()

        layers = re.search(r'deep_linear_(\d+)\.yaml', model).group(1)
        save_dir += '_' + layers +'_hidden'

        f.write(save_dir + "\n")
        f.flush()

        python_cmd = [
            "python", "main.py",
            "--method", method,
            "--data", data,
            "--model", model,
            "--optim", optim,
            "--diagnostics", DIAGNOSTICS,
            "--seed", str(seed),
            "--save_dir", save_dir,
            "--wandb_project", WANDB_PROJECT,
        ]

        if USE_SLURM:
            sbatch_script = dedent(
                f"""\
                #!/bin/bash
                #SBATCH --job-name=cifar_s{seed}
                #SBATCH --output=logs/%j.out
                #SBATCH --error=logs/%j.err
                #SBATCH --gres=gpu:1
                #SBATCH --cpus-per-task=4
                #SBATCH --mem=32GB
                #SBATCH --time=8:00:00

                echo "save_dir: {save_dir}"

                {shlex.join(python_cmd + ['--wandb_not_upload'])}
                """
            )
            subprocess.run(
                ["sbatch"],
                input=sbatch_script,
                text=True,
                check=True,
            )
        else:
            subprocess.run(python_cmd, check=True)

if USE_SLURM:
    print("All jobs submitted. Running Weights & Biases sync daemon. Ctrl+C to stop syncing")
    subprocess.run(
        [
            "python",
            "wandb-sync-daemon.py",
            "--save_dirs",
            str(save_dirs_file),
        ],
        check=True,
    )
else:
    print("All jobs complete.")