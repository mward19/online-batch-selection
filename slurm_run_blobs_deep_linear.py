import copy
from itertools import product
from pathlib import Path
from textwrap import dedent
from datetime import datetime
from tqdm import tqdm
import shlex
import subprocess
import re
import yaml
from scipy.special import ndtr

WANDB_PROJECT = "Matthew—Deep Linear Networks (Blobs) large test"

USE_SLURM = True

SEEDS         = [1, 2, 3]
# SEEDS = [1]
DIAGNOSTICS   = "configs/diagnostics/weight_matrix_tests.yaml"
CONFIG_DIR    = "configs/makeblobs"

DIMS = [32, 1024, 8192]
# DIMS = [8192]
CENTER_SCALES = [0.5, 1.0]
# CENTER_SCALES = [0.5]
METHODS_HYPERPLANE  = ["rholoss-0.1-hyperplane", "bayesian-0.1-hyperplane"]
METHODS_FIXED = [
     f"{CONFIG_DIR}/method/uniform-0.1.yaml",
     f"{CONFIG_DIR}/method/divbs-0.1.yaml",
]
MODEL_CONFIGS = [
    f"{CONFIG_DIR}/model/deep_linear_saxe/deep_linear_1024_3layer.yaml",
    f"{CONFIG_DIR}/model/deep_linear_saxe/deep_linear_1024_16layer.yaml",
    f"{CONFIG_DIR}/model/deep_linear_saxe/deep_linear_1024_64layer.yaml",
    f"{CONFIG_DIR}/model/deep_linear_saxe/deep_linear_8192_3layer.yaml",
]
OPTIMS        = [f"{CONFIG_DIR}/optim/adamw-320-0.001-0.01.yaml"]
GEN_DIR       = Path(CONFIG_DIR) / "generated"


def write_generated_configs(dims, center_scales):
    rholoss_tmpl  = yaml.safe_load(open(f"{CONFIG_DIR}/method/rholoss-0.1-hyperplane-template.yaml"))
    bayesian_tmpl = yaml.safe_load(open(f"{CONFIG_DIR}/method/bayesian-0.1-hyperplane-template.yaml"))
    data_tmpl     = yaml.safe_load(open(f"{CONFIG_DIR}/data/makeblobs-template.yaml"))

    for d, cs in product(dims, center_scales):
        teacher_path = f"models/teacher/makeblobs_{d}d_cscale{cs}_hyperplane_alpha1.0_nseed0.pth"

        for name, tmpl in [
            (f"rholoss-0.1-hyperplane_d{d}_cscale{cs}",  rholoss_tmpl),
            (f"bayesian-0.1-hyperplane_d{d}_cscale{cs}", bayesian_tmpl),
        ]:
            cfg = copy.deepcopy(tmpl)
            cfg['teacher_model_path'] = teacher_path
            p = GEN_DIR / "method" / f"{name}.yaml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(yaml.dump(cfg))

        cfg = copy.deepcopy(data_tmpl)
        cfg['dataset']['n_features']  = d
        cfg['dataset']['input_dim']   = [1, d]
        cfg['dataset']['center_file'] = f"models/teacher/makeblobs_{d}d_cscale{cs}_centers_seed42.npy"
        cfg['dataset']['wstar_file']   = f"models/teacher/makeblobs_{d}d_cscale{cs}_wstar_seed42.npy"
        # alpha1.0 matches the teacher used for RhoLoss/Bayesian in the method configs
        cfg['dataset']['wnoised_file'] = f"models/teacher/makeblobs_{d}d_cscale{cs}_wnoised_alpha1.0_nseed0.npy"
        cfg['bayes_accuracy'] = round(float(ndtr(cs)), 3)
        p = GEN_DIR / "data" / f"makeblobs_d{d}_cscale{cs}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.dump(cfg))


Path("logs").mkdir(exist_ok=True)

for dim, cscale in product(DIMS, CENTER_SCALES):
    print(f"Generating geometry and teacher for dim={dim}, center_scale={cscale}...")
    subprocess.run(
        [
            "python", "data/make_blobs_teacher.py",
            "--n_features", str(dim),
            "--center_scale", str(cscale),
            "--center_seed", "42",
            "--alpha", "1.0",
            "--noise_seed", "0",
            "--out_dir", "models/teacher",
        ],
        check=True,
    )

write_generated_configs(DIMS, CENTER_SCALES)

save_dirs_file = (
    Path("logs")
    / f"save_dirs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

jobs = (
    [
        (
            seed,
            str(GEN_DIR / "data"   / f"makeblobs_d{dim}_cscale{cscale}.yaml"),
            model,
            optim,
            str(GEN_DIR / "method" / f"{method_name}_d{dim}_cscale{cscale}.yaml"),
        )
        for seed, dim, cscale, optim, method_name, model
        in product(SEEDS, DIMS, CENTER_SCALES, OPTIMS, METHODS_HYPERPLANE, MODEL_CONFIGS)
    ] + [
        (
            seed,
            str(GEN_DIR / "data" / f"makeblobs_d{dim}_cscale{cscale}.yaml"),
            model,
            optim,
            method,
        )
        for seed, dim, cscale, optim, method, model
        in product(SEEDS, DIMS, CENTER_SCALES, OPTIMS, METHODS_FIXED, MODEL_CONFIGS)
    ]
)

with open(save_dirs_file, "w") as f:
    for seed, data, model, optim, method in tqdm(
        jobs,
        desc="Submitting jobs" if USE_SLURM else "Running jobs",
        total=len(jobs),
    ):
        save_dir = subprocess.check_output(
            [
                "python",
                "get_save_dir.py",
                "--method", method,
                "--data", data,
                "--model", model,
                "--optim", optim,
                "--seed", str(seed),
            ],
            text=True,
        ).strip()

        model_id = re.search(r'deep_linear_(.+)\.yaml', model).group(1)
        save_dir += f'_{model_id}_hidden'

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
            "--wandb_project", WANDB_PROJECT
        ]

        if USE_SLURM:
            sbatch_script = dedent(
                f"""\
                #!/bin/bash
                #SBATCH --job-name=blobs_s{seed}
                #SBATCH --output=logs/%j.out
                #SBATCH --error=logs/%j.err
                #SBATCH --gres=gpu:1
                #SBATCH --cpus-per-task=4
                #SBATCH --mem=32GB
                #SBATCH --time=1:00:00
                #SBATCH -C pascal

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
