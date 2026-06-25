import yaml
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import secrets
from utils import custom_logger,random_str, get_date, re_nest_configs, get_configs
from run_dir import setup_run_dir, write_guard
import wandb
import json


import torch.multiprocessing as mp
import methods


def build_artifact_stem(args, config):
    # stem_dict = dict(
    #     bsel=config['method'],
    #     seed=config['seed'],
    #     model=config['networks']['type'],
    #     opt=os.path.basename(args.optim).split('-')[0] if args.optim is not None else None,
    #     bs=config['training_opt']['batch_size'],
    #     ratio=config.get('method_opt', {}).get('ratio'),
    #     lr=config['training_opt']['optim_params']['lr'],
    #     wd=config['training_opt']['optim_params']['weight_decay'],
    #     layers=config['networks']['params']['num_hidden_layers'],
    #     hidden_dim=config['networks']['params']['hidden_dim']
    # )
    # if args.artifact_suffix:
    #     stem_dict.update(json.loads(args.artifact_suffix))
    # return json.dumps(stem_dict).replace(' ', '')
    # TODO: change this behavior
    return json.dumps(
        dict(
            bsel=config['method'],
            seed=config['seed'],
            model=config['networks']['type'],
            opt=os.path.basename(args.optim).split('-')[0] if args.optim is not None else None,
            bs=config['training_opt']['batch_size'],
            ratio=config.get('method_opt', {}).get('ratio'),
            lr=config['training_opt']['optim_params']['lr'],
            wd=config['training_opt']['optim_params']['weight_decay'],
            noise_percent=config['dataset'].get('noise_percent', 0.0)
        )
    ).replace(' ', '')


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
    checkpoint_path = os.path.join(run_dir, 'checkpoint.pth.tar')
    if not os.path.isfile(checkpoint_path):
        return None

    checkpoint_preview = _load_checkpoint_preview(checkpoint_path)

    if run_mode == 'extension':
        additional_epochs = training_opt.get('additional_epochs')
        if additional_epochs is not None:
            additional_epochs = int(additional_epochs)
            if additional_epochs < 1:
                raise ValueError('training_opt.additional_epochs must be a positive integer when extending a run.')
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--method',  type=str,
                        default=None,
                        help='batch selection method')
    parser.add_argument('--data',  type=str,
                        default=None,
                        help='dataset name')
    parser.add_argument('--model', type=str,
                        default=None,
                        help='model name')
    parser.add_argument('--optim', type=str,
                        default=None,
                        help='batch size, batch seed, learning rate, optimizer, weight decay')
    parser.add_argument('--diagnostics', type=str,
                        default='configs/diagnostics/default.yaml',
                        help='diagnostics config yaml')
    parser.add_argument('--save_dir', type=str, 
                        default=None,
                        help='directory to save results')
    parser.add_argument('--log_file', type=str, 
                        default=None, 
                        help='Logger file name.')
    parser.add_argument('--notes', type=str,
                        default=None, 
                        help='Notes for the experiment.')
    parser.add_argument('--wandb_not_upload', action='store_true', 
                        help='Do not upload the result to wandb.')
    parser.add_argument('--wandb_project', type=str,
                        default=None, help='Project name for W&B')
    parser.add_argument('--artifact_suffix', type=str, default=None,
                        help='JSON-encoded dict of extra fields merged into artifact_stem for snapshot/selected-points file names.')
    parser.add_argument('--exp_base', type=str, default='./exp/',
                        help='Base directory for experiment outputs; also used to namespace the snapshots dir.')

    args = parser.parse_args()

    # ============================================================================
    # load config file
    print('=====> Loading config files: \n' + args.method + '\n' + args.data + '\n' + args.model + '\n' + args.optim + '\n' + args.diagnostics)
    method_config = get_configs(args.method)
    data_config = get_configs(args.data)
    model_config = get_configs(args.model)
    optim_config = get_configs(args.optim)
    diagnostics_config = get_configs(args.diagnostics)
    config = {**method_config, **data_config, **model_config, **optim_config, **diagnostics_config} # combine into single config
    config['seed'] = args.seed # add seed to config
    config['artifact_stem'] = build_artifact_stem(args, config)
    print('=====> Config files loaded.')

    if args.log_file is not None:
        config['log_file'] = args.log_file

    training_opt = config.setdefault('training_opt', {})
    resume_from = training_opt.get('resume_run_path') or None
    run_dir, run_mode, run_info = setup_run_dir(resume_from=resume_from)

    # method/save_dir
    save_dir = run_dir
    config['save_dir'] = save_dir
    config['exp_base'] = args.exp_base
    method = config['method']

    if method not in methods.__all__:
        raise ValueError(f'Method {method} is not supported. Please check the methods.py file.')

    resume_state = _configure_resume_state(run_mode, run_dir, run_info, config)


    # wandb_not_upload
    if args.wandb_not_upload:
        os.environ["WANDB_MODE"] = "dryrun"
    else:
        os.environ["WANDB_MODE"] = "run"

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
        config['resume'] = {'from': run_info['parent_dir']}
    if run_mode != 'restart':
        logger.info('=====> Saving config file')
        if run_mode == 'fresh':
            write_guard(config_path)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info('=====> Config file saved')


    init_seeds(args.seed)
    # logger.info(f'=====> Random seed initialized to {config["seed"]}')
    logger.info(f'=====> Wandb initialized')
    wandb_init_kwargs = {
        'config': config,
        'project': args.wandb_project,
        'dir': save_dir,
    }
    if resume_state is not None:
        if resume_state['wandb_run_id'] is None:
            raise ValueError(
                f"Unable to determine the W&B run id for resumed run at '{run_dir}'. "
                "Expected wandb_run_id.txt, checkpoint metadata, or wandb_local_path.txt."
            )
        wandb_init_kwargs['id'] = resume_state['wandb_run_id']
        wandb_init_kwargs['resume'] = 'must' if run_mode == 'extension' else 'allow'
    run = wandb.init(**wandb_init_kwargs)
    re_nest_configs(run.config)
    wandb.define_metric('acc', 'max')
    if resume_state is None:
        if 'noise_percent' in config['dataset'].keys():
            run.name = (
                f"{method}_{config['dataset']['name']}_"
                f"{config['dataset']['noise_percent']}p_"
                f"{config['training_opt']['optimizer']}_Seed{config['seed']}"
            )
        else:
            run.name = (
                f"{method}_{config['dataset']['name']}_"
                f"{config['training_opt']['optimizer']}_Seed{config['seed']}"
            )
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