"""Submit the CIFAR3 deep-linear sweep using templated single-file configs.

Configs are generated from a template into ./configs-temp/ and each is run via
`main.py --config <generated>` (seed is a swept top-level config key, not a CLI
flag). Run output dirs are claimed at runtime under ./experiments/, so they
are no longer precomputed here; SLURM stdout/stderr go to logs/slurm/%j.{out,err}.
Jobs request --requeue so preemption restarts land back in the same run dir.
"""

import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_configs import generate_configs
from utils import run_job, RunType

RUN_TYPE = RunType.SBATCH

CONFIG = "template_configs/mnist_basic.yaml"

Path("logs/slurm").mkdir(parents=True, exist_ok=True)

# Download the CLIP teacher on the login node before any compute job runs.
# subprocess.run(["python", "perform_downloads.py", "--method", config_path], check=True)

run_job(CONFIG, RUN_TYPE)

print("Completed.")
