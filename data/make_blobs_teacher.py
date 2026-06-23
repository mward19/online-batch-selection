import argparse
import math
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description='Generate cluster geometry and Linear teacher model for MakeBlobs experiments.'
    )
    parser.add_argument('--n_features', type=int, default=1024)
    parser.add_argument('--center_scale', type=float, default=1.0,
                        help='Scale of cluster centers; centers placed at ±center_scale * w*')
    parser.add_argument('--center_seed', type=int, default=42,
                        help='RNG seed for generating the ground-truth direction w*')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dimensionless noise level; noise_std = alpha / sqrt(n_features). '
                             'alpha=1 gives theta=45 deg between teacher and w*.')
    parser.add_argument('--noise_seed', type=int, default=0,
                        help='RNG seed for the noise added to w*')
    parser.add_argument('--out_dir', type=str, default='models/teacher')
    args = parser.parse_args()

    d = args.n_features
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_tag = f'alpha{args.alpha}'
    cscale_tag = f'cscale{args.center_scale}'

    # 1. Generate ground-truth direction w*
    rng = np.random.default_rng(args.center_seed)
    w_star = rng.standard_normal(d).astype(np.float32)
    w_star /= np.linalg.norm(w_star)

    # 2. Save cluster centers (shape 2 x d) and w*
    centers = np.stack([+args.center_scale * w_star, -args.center_scale * w_star])
    centers_path = out_dir / f'makeblobs_{d}d_{cscale_tag}_centers_seed{args.center_seed}.npy'
    wstar_path   = out_dir / f'makeblobs_{d}d_{cscale_tag}_wstar_seed{args.center_seed}.npy'
    np.save(centers_path, centers)
    np.save(wstar_path, w_star)
    print(f'Saved centers : {centers_path}')
    print(f'Saved w*      : {wstar_path}')

    # 3. Generate noised direction w_hat
    noise_std = args.alpha / math.sqrt(d)
    rng_noise = np.random.default_rng(args.noise_seed)
    eps = rng_noise.standard_normal(d).astype(np.float32)
    w_noised = w_star + noise_std * eps
    w_hat = w_noised / np.linalg.norm(w_noised)

    cos_theta = float(np.dot(w_star, w_hat))
    theta_deg = math.degrees(math.acos(min(1.0, abs(cos_theta))))
    print(f'alpha={args.alpha}, noise_std={noise_std:.5f}, theta={theta_deg:.1f} deg')

    # 4. Save w_hat and teacher state dict
    wnoised_path = out_dir / f'makeblobs_{d}d_{cscale_tag}_wnoised_{alpha_tag}_nseed{args.noise_seed}.npy'
    teacher_path = out_dir / f'makeblobs_{d}d_{cscale_tag}_hyperplane_{alpha_tag}_nseed{args.noise_seed}.pth'
    np.save(wnoised_path, w_hat)

    w_hat_t = torch.from_numpy(w_hat)
    state_dict = {
        'fc.weight': torch.stack([+w_hat_t, -w_hat_t]),
        'fc.bias':   torch.zeros(2),
    }
    torch.save(state_dict, teacher_path)
    print(f'Saved w_hat   : {wnoised_path}')
    print(f'Saved teacher : {teacher_path}')


if __name__ == '__main__':
    main()
