import yaml
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import secrets
from utils import custom_logger,random_str, get_date, re_nest_configs, get_configs, get_save_dir
import wandb
import json


import torch.multiprocessing as mp
import methods


def build_artifact_stem(args, config):
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
            layers=config['networks']['params']['num_hidden_layers'],
            hidden_dim=config['networks']['params']['hidden_dim']
        )
    ).replace(' ', '')


def _normalize_path(path):
    return os.path.abspath(os.path.expanduser(path))


def _default_resume_checkpoint_path(resume_run_path):
    return os.path.join(resume_run_path, 'checkpoint.pth.tar')


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


def _configure_resume_state(args, config):
    training_opt = config.setdefault('training_opt', {})
    resume_run_path = training_opt.get('resume_run_path')
    if resume_run_path is None or str(resume_run_path).strip() == '':
        return None

    resume_run_path = _normalize_path(resume_run_path)
    if args.save_dir is not None and _normalize_path(args.save_dir) != resume_run_path:
        raise ValueError(
            f"save_dir '{args.save_dir}' must match training_opt.resume_run_path '{resume_run_path}' when resuming a run."
        )

    checkpoint_path = training_opt.get('resume')
    if checkpoint_path is None or str(checkpoint_path).strip() == '':
        checkpoint_path = _default_resume_checkpoint_path(resume_run_path)
    else:
        checkpoint_path = _normalize_path(checkpoint_path)

    checkpoint_preview = _load_checkpoint_preview(checkpoint_path)
    additional_epochs = training_opt.get('additional_epochs')
    if additional_epochs is not None:
        additional_epochs = int(additional_epochs)
        if additional_epochs < 1:
            raise ValueError('training_opt.additional_epochs must be a positive integer when resuming a run.')
        training_opt['num_epochs'] = int(checkpoint_preview['epoch']) + additional_epochs

    training_opt['resume_run_path'] = resume_run_path
    training_opt['resume'] = checkpoint_path
    args.save_dir = resume_run_path

    return {
        'resume_run_path': resume_run_path,
        'checkpoint_path': checkpoint_path,
        'checkpoint_preview': checkpoint_preview,
        'wandb_run_id': _resolve_wandb_run_id(resume_run_path, checkpoint_preview),
    }



def init_seeds(seed):
    print('=====> Using fixed random seed: ' + str(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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

    resume_state = _configure_resume_state(args, config)




    if args.log_file is not None:
        config['log_file'] = args.log_file
    

    if args.save_dir is None:
        args.save_dir = get_save_dir(config, args.notes)
    

    # method/save_dir
    save_dir = args.save_dir
    config['save_dir'] = save_dir
    method = config['method']

    if method not in methods.__all__:
        raise ValueError(f'Method {method} is not supported. Please check the methods.py file.')

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)


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


    # save config file
    logger.info('=====> Saving config file')
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
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
                f"Unable to determine the W&B run id for resumed run at '{resume_state['resume_run_path']}'. "
                "Expected wandb_run_id.txt, checkpoint metadata, or wandb_local_path.txt."
            )
        wandb_init_kwargs['id'] = resume_state['wandb_run_id']
        wandb_init_kwargs['resume'] = 'must'
    run = wandb.init(**wandb_init_kwargs)
    re_nest_configs(run.config)
    wandb.define_metric('acc', 'max')
    if resume_state is None:
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

    wandb_local_path = wandb.run.dir
    # save wandb_local_path to wandb_local_path.txt
    with open(os.path.join(save_dir, 'wandb_local_path.txt'), 'w') as f:
        f.write(wandb_local_path)
        f.close()
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