"""Submit the MNIST sweep using templated single-file configs (§4.2).

Configs are generated from a template into ./configs-temp/ and each is run via
`main.py --config <generated>` (seed is a swept top-level config key, not a CLI
flag). Run output dirs are claimed at runtime under ./experiments/ (§2), so they
are no longer precomputed here; SLURM stdout/stderr go to logs/slurm/%j.{out,err}.
Jobs request --requeue so preemption restarts land back in the same run dir (§9.2).
"""

from pathlib import Path
from textwrap import dedent
import shlex
import subprocess

from generate_configs import generate_configs

USE_SLURM = True

# TEMPLATE = "configs/mnist_basic.yaml"

# Cartesian product over these fills the template's __REQUIRED__ leaves (incl. seed).
PARAMS_TO_VARY = {
    "seed": [10],
    "method": ["RhoLoss"],
    "training_opt.optim_params.lr": [0.1, 0.01],
    "dataset.noise_percent": [0.0, 0.2, 0.8]
}

# config_paths = generate_configs(TEMPLATE, PARAMS_TO_VARY)
config_paths = ["configs/todd_stalled_run.yaml"]
Path("logs/slurm").mkdir(parents=True, exist_ok=True)

for config_path in config_paths:
    # Download the CLIP teacher on the login node before any compute job runs.
    subprocess.run(["python", "perform_downloads.py", "--method", config_path], check=True)

    python_cmd = [
        "python", "main.py",
        "--config", config_path,
        "--wandb_not_upload",
    ]

    if USE_SLURM:
        sbatch_script = dedent(
            f"""\
            #!/bin/bash
            #SBATCH --job-name=mnist
            #SBATCH --output=logs/slurm/%j.out
            #SBATCH --error=logs/slurm/%j.err
            #SBATCH --gres=gpu:1
            #SBATCH --cpus-per-task=4
            #SBATCH --mem=16GB
            #SBATCH --time=1:00:00
            #SBATCH --requeue
            #SBATCH --qos=standby

            {shlex.join(python_cmd)}
            """
        )
        subprocess.run(["sbatch"], input=sbatch_script, text=True, check=True)
    else:
        subprocess.run(python_cmd, check=True)

print("All jobs submitted." if USE_SLURM else "All jobs complete.")
