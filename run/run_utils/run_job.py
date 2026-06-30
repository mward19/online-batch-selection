from pathlib import Path
import shlex
from tqdm import tqdm
import subprocess
from enum import Enum

class RunType(Enum):
    DRY = 0 # Does not run at all
    NORMAL = 1 # For use on systems without slurm
    SBATCH = 2 # To run jobs in the background
    SRUN = 3   # To run jobs with slurm in series, not in the background

def run_job(
        config_path, 
        run_type: RunType = RunType.SBATCH,
        *,
        cpus: str = '4',
        mem: str = '32GB', # gb
        time: str = '1:00:00',
        name: str = 'online-bs',
        preemptible=True,
        download=True
    ):
    if download:
        download_cmd = ["python", "perform_downloads.py", "--method", config_path]
        if run_type == RunType.DRY:
            tqdm.write(f'Dry run. Would have run `{download_cmd}`')    
        else:
            # Perform necessary downloads
            subprocess.run(download_cmd, check=True)

    python_cmd = ["python", "main.py", "--config", config_path]

    if run_type == RunType.DRY:
        tqdm.write(f'Dry run. Would have run `{python_cmd}`')

    if run_type == RunType.NORMAL:
        subprocess.run(python_cmd, check=True)
        return
    
    # Compute nodes don't have internet
    python_cmd.append("--wandb_not_upload")

    slurm_flags = [
        "--gres=gpu:1",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--time={time}",
        f"--job-name={name}",
    ]

    if run_type == RunType.SRUN:
        subprocess.run(["srun"] + slurm_flags + python_cmd, check=True)
        return
    
    # Prepare slurm log dir
    Path("logs/slurm").mkdir(parents=True, exist_ok=True)
    slurm_flags += [
        "--output=logs/slurm/%j.out",
        "--error=logs/slurm/%j.err",
    ]

    if preemptible:
        slurm_flags += [
            "--requeue",
            "--qos=standby"
        ]


    if run_type == RunType.SBATCH:
        subprocess.run(
            ["sbatch"] + slurm_flags + [f"--wrap={shlex.join(python_cmd)}"],
            check=True,
        )
        return