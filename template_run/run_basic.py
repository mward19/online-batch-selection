"""Submit the basic single-dataset baseline runs.

Each entry is a concrete merged config under ./config_templates/ run via
`main.py --config <config>` (the seed is a top-level key in the config). Run
output dirs are claimed at runtime under ./experiments/; SLURM stdout/stderr
go to logs/slurm/%j.{out,err}. Jobs request --requeue so preemption restarts land
back in the same run dir.
"""

import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import run_job, RunType

RUN_TYPE = RunType.SBATCH

# (config, job-name tag, wall-time, memory). Synthetic/light jobs ask for less.
RUNS = [
    ("template_configs/makeblobs_basic.yaml",          "makeblobs", "1:00:00", "8GB"),
    ("template_configs/teacher_generated_basic.yaml",  "teachergen", "2:00:00", "8GB"),
    ("template_configs/mnist_basic.yaml",              "mnist",     "4:00:00", "16GB"),
    ("template_configs/cifar3_basic.yaml",             "cifar3",    "8:00:00", "32GB"),
    ("template_configs/cifar10_basic.yaml",            "cifar10",   "8:00:00", "32GB"),
]

Path("logs/slurm").mkdir(parents=True, exist_ok=True)

for config_path, tag, walltime, mem in tqdm(RUNS, desc="Submitting jobs"):
    run_job(config_path, RUN_TYPE, time=walltime, mem=mem, name=tag)

print("Completed.")
