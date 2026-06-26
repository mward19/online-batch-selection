"""Submit the basic single-dataset baseline runs.

Each entry is a concrete merged config (§4.1) under ./configs/ run via
`main.py --config <config>` (the seed is a top-level key in the config). Run
output dirs are claimed at runtime under ./experiments/ (§2); SLURM stdout/stderr
go to logs/slurm/%j.{out,err}. Jobs request --requeue so preemption restarts land
back in the same run dir (§9.2). Set USE_SLURM=False to run locally instead.
"""

from pathlib import Path
from textwrap import dedent
import shlex
import subprocess

USE_SLURM = True

# (config, job-name tag, wall-time, memory). Synthetic/light jobs ask for less.
RUNS = [
    ("configs/makeblobs_basic.yaml",          "makeblobs", "1:00:00", "8GB"),
    ("configs/teacher_generated_basic.yaml",  "teachergen", "2:00:00", "8GB"),
    ("configs/mnist_basic.yaml",              "mnist",     "4:00:00", "16GB"),
    ("configs/cifar3_basic.yaml",             "cifar3",    "8:00:00", "32GB"),
    ("configs/cifar10_basic.yaml",            "cifar10",   "8:00:00", "32GB"),
]

Path("logs/slurm").mkdir(parents=True, exist_ok=True)

for config_path, tag, walltime, mem in RUNS:
    python_cmd = [
        "python", "main.py",
        "--config", config_path,
    ]

    if USE_SLURM:
        sbatch_script = dedent(
            f"""\
            #!/bin/bash
            #SBATCH --job-name={tag}
            #SBATCH --output=logs/slurm/%j.out
            #SBATCH --error=logs/slurm/%j.err
            #SBATCH --gres=gpu:1
            #SBATCH --cpus-per-task=4
            #SBATCH --mem={mem}
            #SBATCH --time={walltime}
            #SBATCH --requeue

            {shlex.join(python_cmd)}
            """
        )
        subprocess.run(["sbatch"], input=sbatch_script, text=True, check=True)
    else:
        subprocess.run(python_cmd, check=True)

print("All jobs submitted." if USE_SLURM else "All jobs complete.")
