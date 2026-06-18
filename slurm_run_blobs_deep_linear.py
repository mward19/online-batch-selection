import copy
from itertools import product
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import shlex
import subprocess
import re
import yaml
from scipy.special import ndtr

WANDB_PROJECT = "Matthew—Deep Linear Networks (Ablation June 15)"

USE_SLURM = True

SEEDS         = [1]
DIAGNOSTICS   = "configs/diagnostics/weight_matrix_tests.yaml"
CONFIG_DIR    = "configs/makeblobs"

DIMS = [32]
CENTER_SCALES = [1.0, 3.5]
# N_SAMPLES = [32, 128, 1024, 16384]
N_SAMPLES = [16384]
ALPHAS = [1.5]  # noise_std = alpha / sqrt(n_features)
EXP_BASE = "./exp-ablation/" 
# METHODS_HYPERPLANE  = ["rholoss-0.1-hyperplane", "bayesian-0.1-hyperplane"]
METHODS_HYPERPLANE  = [] # ["rholoss-0.1-hyperplane"]
METHODS_FIXED = [
     f"{CONFIG_DIR}/method/uniform-0.1.yaml",
#      f"{CONFIG_DIR}/method/divbs-0.1.yaml",
]
# METHODS_FIXED = []
MODEL_CONFIGS = [
    # f"{CONFIG_DIR}/model/deep_linear_saxe/deep_linear_1024_3layer.yaml",
    f"{CONFIG_DIR}/model/deep_linear_saxe/deep_linear_1024_16layer.yaml",
    f"{CONFIG_DIR}/model/deep_linear_saxe/deep_linear_relu_1024_16layer.yaml",
    # f"{CONFIG_DIR}/model/deep_linear_saxe/deep_linear_1024_64layer.yaml",
]
OPTIMS = [
    f"{CONFIG_DIR}/optim/adamw.yaml",
    # f"{CONFIG_DIR}/optim/sgd-step1.yaml",
    # f"{CONFIG_DIR}/optim/sgd-step0.1.yaml",
    f"{CONFIG_DIR}/optim/sgd-step0.01.yaml",
    # f"{CONFIG_DIR}/optim/sgd-step0.0001.yaml",
    # f"{CONFIG_DIR}/optim/sgd-step0.00001.yaml",
]
GEN_DIR       = Path(CONFIG_DIR) / "generated"


def make_sbatch(
    cmd: list[str],
    job_name: str,
    time: str = "1:00:00",
    dependency: str | None = None,
    save_dir: str | None = None,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
    ]
    if dependency:
        lines.append(f"#SBATCH --dependency=afterok:{dependency}")
    lines += [
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=32GB",
        f"#SBATCH --time={time}",
        "#SBATCH -C pascal",
        "",
    ]
    if save_dir:
        lines.append(f'echo "save_dir: {save_dir}"')
        lines.append("")
    lines.append(shlex.join(cmd))
    return "\n".join(lines) + "\n"


def write_generated_configs(dims, center_scales, n_samples_list, alphas):
    rholoss_tmpl  = yaml.safe_load(open(f"{CONFIG_DIR}/method/rholoss-0.1-hyperplane-template.yaml"))
    bayesian_tmpl = yaml.safe_load(open(f"{CONFIG_DIR}/method/bayesian-0.1-hyperplane-template.yaml"))
    data_tmpl     = yaml.safe_load(open(f"{CONFIG_DIR}/data/makeblobs-template.yaml"))

    for d, cs, alpha in product(dims, center_scales, alphas):
        teacher_path = f"models/teacher/makeblobs_{d}d_cscale{cs}_hyperplane_alpha{alpha}_nseed0.pth"

        for name, tmpl in [
            (f"rholoss-0.1-hyperplane_d{d}_cscale{cs}_alpha{alpha}",  rholoss_tmpl),
            (f"bayesian-0.1-hyperplane_d{d}_cscale{cs}_alpha{alpha}", bayesian_tmpl),
        ]:
            cfg = copy.deepcopy(tmpl)
            cfg['teacher_model_path'] = teacher_path
            p = GEN_DIR / "method" / f"{name}.yaml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(yaml.dump(cfg))

        base_cfg = copy.deepcopy(data_tmpl)
        base_cfg['dataset']['n_features']  = d
        base_cfg['dataset']['input_dim']   = [1, d]
        base_cfg['dataset']['random_state'] = 42
        base_cfg['dataset']['center_file'] = f"models/teacher/makeblobs_{d}d_cscale{cs}_centers_seed42.npy"
        base_cfg['dataset']['wstar_file']  = f"models/teacher/makeblobs_{d}d_cscale{cs}_wstar_seed42.npy"
        base_cfg['dataset']['wnoised_file'] = f"models/teacher/makeblobs_{d}d_cscale{cs}_wnoised_alpha{alpha}_nseed0.npy"
        base_cfg['bayes_accuracy'] = round(float(ndtr(cs)), 3)

        for n in n_samples_list:
            cfg = copy.deepcopy(base_cfg)
            cfg['dataset']['n_samples'] = n
            p = GEN_DIR / "data" / f"makeblobs_d{d}_cscale{cs}_n{n}_alpha{alpha}.yaml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(yaml.dump(cfg))


Path("logs").mkdir(exist_ok=True)

for dim, cscale, alpha in product(DIMS, CENTER_SCALES, ALPHAS):
    print(f"Generating geometry and teacher for dim={dim}, center_scale={cscale}, alpha={alpha}...")
    subprocess.run(
        [
            "python", "data/make_blobs_teacher.py",
            "--n_features", str(dim),
            "--center_scale", str(cscale),
            "--center_seed", "42",
            "--alpha", str(alpha),
            "--noise_seed", "0",
            "--out_dir", "models/teacher",
        ],
        check=True,
    )

write_generated_configs(DIMS, CENTER_SCALES, N_SAMPLES, ALPHAS)

labels_job_ids: dict[tuple, str] = {}  # (dim, cscale, n) -> slurm job id
# Labels are alpha-independent (determined by centers + random_state, not wnoised),
# so one job per (dim, cscale, n) suffices for all alphas.
for dim, cscale, n in product(DIMS, CENTER_SCALES, N_SAMPLES):
    data_cfg = str(GEN_DIR / "data" / f"makeblobs_d{dim}_cscale{cscale}_n{n}_alpha{ALPHAS[0]}.yaml")
    out_path = f"labels/makeblobs_d{dim}_cscale{cscale}_n{n}.p"
    cmd = [
        "python", "save_labels.py",
        "--data", data_cfg,
        "--output", out_path,
        "--overwrite",
    ]
    if USE_SLURM:
        result = subprocess.run(
            ["sbatch"],
            input=make_sbatch(cmd, f"labels_d{dim}_cs{cscale}_n{n}", time="0:15:00"),
            text=True,
            check=True,
            capture_output=True,
        )
        labels_job_ids[(dim, cscale, n)] = result.stdout.strip().split()[-1]
    else:
        subprocess.run(cmd, check=True)

save_dirs_file = (
    Path("logs")
    / f"save_dirs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

jobs = (
    [
        (
            seed, dim, cscale, n, alpha,
            str(GEN_DIR / "data"   / f"makeblobs_d{dim}_cscale{cscale}_n{n}_alpha{alpha}.yaml"),
            model,
            optim,
            str(GEN_DIR / "method" / f"{method_name}_d{dim}_cscale{cscale}_alpha{alpha}.yaml"),
        )
        for seed, dim, cscale, n, alpha, optim, method_name, model
        in product(SEEDS, DIMS, CENTER_SCALES, N_SAMPLES, ALPHAS, OPTIMS, METHODS_HYPERPLANE, MODEL_CONFIGS)
    ] + [
        (
            seed, dim, cscale, n, alpha,
            str(GEN_DIR / "data" / f"makeblobs_d{dim}_cscale{cscale}_n{n}_alpha{alpha}.yaml"),
            model,
            optim,
            method,
        )
        for seed, dim, cscale, n, alpha, optim, method, model
        in product(SEEDS, DIMS, CENTER_SCALES, N_SAMPLES, ALPHAS, OPTIMS, METHODS_FIXED, MODEL_CONFIGS)
    ]
)

with open(save_dirs_file, "w") as f:
    for seed, dim, cscale, n, alpha, data, model, optim, method in tqdm(
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
                "--exp_base", EXP_BASE,
            ],
            text=True,
        ).strip()

        save_dir += f'_d{dim}_cscale{cscale}_n{n}_alpha{alpha}'
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
            "--wandb_project", WANDB_PROJECT,
            "--artifact_suffix", f"d{dim}_cscale{cscale}_n{n}_alpha{alpha}",
        ]

        if USE_SLURM:
            dep = labels_job_ids.get((dim, cscale, n))
            sbatch_script = make_sbatch(
                python_cmd + ['--wandb_not_upload'],
                job_name=f"blobs_s{seed}",
                dependency=dep,
                save_dir=save_dir,
                time='2:00:00'
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
