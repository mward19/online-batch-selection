"""Submit the CIFAR3 deep-linear sweep using templated single-file configs (§4.2).

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

TEMPLATE = "configs/cifar3_deep_linear_template.yaml"

# Cartesian product over these fills the template's __REQUIRED__ leaves (incl. seed).
PARAMS_TO_VARY = {
    "seed": [1],
    "method": ["RhoLoss"],
    "networks.params.num_hidden_layers": [3],
}

config_paths = generate_configs(TEMPLATE, PARAMS_TO_VARY)
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
            #SBATCH --job-name=cifar3
            #SBATCH --output=logs/slurm/%j.out
            #SBATCH --error=logs/slurm/%j.err
            #SBATCH --gres=gpu:1
            #SBATCH --cpus-per-task=4
            #SBATCH --mem=32GB
            #SBATCH --time=8:00:00
            #SBATCH --requeue

            {shlex.join(python_cmd)}
            """
        )
        subprocess.run(["sbatch"], input=sbatch_script, text=True, check=True)
    else:
        subprocess.run(python_cmd, check=True)

print("All jobs submitted." if USE_SLURM else "All jobs complete.")
