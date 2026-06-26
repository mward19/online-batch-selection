import yaml
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import secrets
from utils import custom_logger,random_str, get_date, re_nest_configs, get_configs
from run_dir import setup_run_dir, write_guard, build_run_name
import wandb


import torch.multiprocessing as mp
import methods


def _load_checkpoint_preview(checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"Resume checkpoint not found at '{checkpoint_path}'.")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'epoch' not in checkpoint:
        raise ValueError(f"Resume checkpoint at '{checkpoint_path}' is missing the saved epoch.")
    return checkpoint


def _parse_wandb_run_id_from_local_path(wandb_local_path):
    if wandb_local_path is None:
        return None

    run_dir = os.path.basename(os.path.dirname(str(wandb_local_path).rstrip(os.sep)))
    if '-' not in run_dir:
        return None
    return run_dir.rsplit('-', 1)[-1] or None


def _resolve_wandb_run_id(resume_run_path, checkpoint_preview):
    explicit_run_id_path = os.path.join(resume_run_path, 'wandb_run_id.txt')
    if os.path.isfile(explicit_run_id_path):
        with open(explicit_run_id_path, 'r') as f:
            run_id = f.read().strip()
        if run_id:
            return run_id

    checkpoint_run_id = checkpoint_preview.get('wandb_run_id')
    if checkpoint_run_id:
        return str(checkpoint_run_id)

    wandb_local_path_file = os.path.join(resume_run_path, 'wandb_local_path.txt')
    if os.path.isfile(wandb_local_path_file):
        with open(wandb_local_path_file, 'r') as f:
            wandb_local_path = f.read().strip()
        run_id = _parse_wandb_run_id_from_local_path(wandb_local_path)
        if run_id:
            return run_id

    return None


def _configure_resume_state(run_mode, run_dir, run_info, config):
    """Wire up resume after the run dir has been resolved (§9).

    ``extension`` and ``restart`` both read their checkpoint from the (now
    local) run dir; extension reattaches the parent W&B run, restart reattaches
    its own. A restart that requeued before the first checkpoint starts fresh in
    place.
    """
    if run_mode == 'fresh':
        return None

    training_opt = config['training_opt']
    checkpoint_path = os.path.join(run_dir, 'snapshots', 'checkpoint.pth.tar')
    if not os.path.isfile(checkpoint_path):
        return None

    checkpoint_preview = _load_checkpoint_preview(checkpoint_path)

    if run_mode == 'extension':
        additional_epochs = config.get('resume', {}).get('additional_epochs')
        if additional_epochs is not None:
            additional_epochs = int(additional_epochs)
            if additional_epochs < 1:
                raise ValueError('resume.additional_epochs must be a positive integer when extending a run.')
            training_opt['num_epochs'] = int(checkpoint_preview['epoch']) + additional_epochs

    training_opt['resume'] = checkpoint_path

    return {
        'run_mode': run_mode,
        'checkpoint_path': checkpoint_path,
        'checkpoint_preview': checkpoint_preview,
        'wandb_run_id': _resolve_wandb_run_id(run_dir, checkpoint_preview),
    }



def init_seeds(seed):
    print('=====> Using fixed random seed: ' + str(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def main():
    # ============================================================================
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='single merged config YAML (§4.1)')
    parser.add_argument('--log_file', type=str,
                        default=None,
                        help='Logger file name.')
    parser.add_argument('--notes', type=str,
                        default=None,
                        help='Notes for the experiment.')
    parser.add_argument('--wandb_not_upload', action='store_true',
                        help='Do not upload the result to wandb.')
    parser.add_argument('--experiments_dir', type=str, default='./experiments',
                        help='Base directory under which run directories are created.')

    args = parser.parse_args()

    # ============================================================================
    # load the single merged config file (§4.1)
    print(f'=====> Loading config: {args.config}')
    config = get_configs(args.config)
    if 'seed' not in config:
        raise ValueError("'seed' is required as a top-level config key but was not provided.")
    config['seed'] = int(config['seed'])
    config['run_name'] = build_run_name(config, config.get('run_name_format'))
    print(f"=====> Config loaded. Run name: {config['run_name']}")

    if args.log_file is not None:
        config['log_file'] = args.log_file

    config.setdefault('training_opt', {})
    resume_from = config.get('resume', {}).get('from') or None
    run_dir, run_mode, run_info = setup_run_dir(
        config['run_name'], experiments_root=args.experiments_dir, resume_from=resume_from)

    # method/save_dir
    save_dir = run_dir
    config['save_dir'] = save_dir
    method = config['method']

    if method not in methods.__all__:
        raise ValueError(f'Method {method} is not supported. Please check the methods.py file.')

    resume_state = _configure_resume_state(run_mode, run_dir, run_info, config)


    # W&B init kwargs come from the config's wandb section (§4.1).
    wandb_kwargs = dict(config.get('wandb', {}))
    if args.wandb_not_upload:
        os.environ["WANDB_MODE"] = "dryrun"
        wandb_kwargs.pop('mode', None)

    if args.log_file is None:
        logger = custom_logger(save_dir)
    else:
        logger = custom_logger(save_dir, args.log_file)

    logger.info('========================= Start Main =========================')
    logger.info(f'=====> Run directory ({run_mode}): {run_dir}')


    # save config file (fresh: guarded write; extension: refresh the copied
    # snapshot with the updated epoch budget + lineage; restart: keep existing)
    config_path = os.path.join(save_dir, 'config.yaml')
    if run_mode == 'extension':
        config.setdefault('resume', {})['from'] = run_info['parent_dir']
    if run_mode != 'restart':
        logger.info('=====> Saving config file')
        if run_mode == 'fresh':
            write_guard(config_path)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info('=====> Config file saved')


    init_seeds(config['seed'])
    # logger.info(f'=====> Random seed initialized to {config["seed"]}')
    logger.info(f'=====> Wandb initialized')
    wandb_kwargs['config'] = config
    wandb_kwargs['dir'] = save_dir
    if resume_state is not None:
        if resume_state['wandb_run_id'] is None:
            raise ValueError(
                f"Unable to determine the W&B run id for resumed run at '{run_dir}'. "
                "Expected wandb_run_id.txt, checkpoint metadata, or wandb_local_path.txt."
            )
        wandb_kwargs['id'] = resume_state['wandb_run_id']
        wandb_kwargs['resume'] = 'must' if run_mode == 'extension' else 'allow'
    run = wandb.init(**wandb_kwargs)
    re_nest_configs(run.config)
    wandb.define_metric('acc', 'max')
    if resume_state is None:
        run.name = config['run_name']
    else:
        logger.info(
            f"=====> Resuming W&B run {run.id} from {resume_state['checkpoint_path']} "
            f"through epoch {config['training_opt']['num_epochs']}"
        )

    config['wandb_run_id'] = run.id

    if run_mode == 'fresh':
        # save wandb_local_path to wandb_local_path.txt
        wandb_local_path = wandb.run.dir
        with open(os.path.join(save_dir, 'wandb_local_path.txt'), 'w') as f:
            f.write(wandb_local_path)
        with open(os.path.join(save_dir, 'wandb_run_id.txt'), 'w') as f:
            f.write(run.id)

    config['num_gpus'] = torch.cuda.device_count()
    logger.info(f'=====> Number of GPUs: {config["num_gpus"]}')

    Method = getattr(methods, method)(config, logger)
    Method.run()

    logger.info('========================= End Main =========================')

    logger.wandb_finish()



if __name__ == '__main__':
    main()