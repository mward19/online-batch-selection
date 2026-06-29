"""Submit the makeblobs antipodal sweep using templated single-file configs.

Configs are generated from a template into ./configs-temp/ and each is run via
`main.py --config <generated>` (seed is a swept top-level config key, not a CLI
flag). Run output dirs are claimed at runtime under ./experiments/, so they
are no longer precomputed here; SLURM stdout/stderr go to logs/slurm/%j.{out,err}.
Jobs request --requeue so preemption restarts land back in the same run dir.
"""

import os
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_configs import generate_configs
from utils import run_job, RunType


RUN_TYPE = RunType.SBATCH
Path("logs/slurm").mkdir(parents=True, exist_ok=True)

# RHOLOSS with scores used as distribution ------------------------------------------
TEMPLATE = "configs/makeblobs_antipodal_dist.yaml"

# Cartesian product over these fills the template's __REQUIRED__ leaves (incl. seed).
PARAMS_TO_VARY = {
    "seed": [1, 2, 3],
    "method_opt.softmax_lambda": [0.001, 0.01],
}

config_paths = generate_configs(TEMPLATE, PARAMS_TO_VARY)

for config_path in tqdm(config_paths, desc="Submitting jobs"):
    subprocess.run(["python", "perform_downloads.py", "--method", config_path], check=True)
    run_job(config_path, RUN_TYPE)

# Normal RHOLOSS --------------------------------------------------------------------
# RUN_TYPE = RunType.SBATCH
# TEMPLATE = "configs/makeblobs_antipodal_standard.yaml"

# # Cartesian product over these fills the template's __REQUIRED__ leaves (incl. seed).
# PARAMS_TO_VARY = {
#     "seed": [1, 2, 3],
# }

# config_paths = generate_configs(TEMPLATE, PARAMS_TO_VARY)

# for config_path in tqdm(config_paths, desc="Submitting jobs"):
#     subprocess.run(["python", "perform_downloads.py", "--method", config_path], check=True)
#     run_job(config_path, RUN_TYPE)

# print("Completed.")
