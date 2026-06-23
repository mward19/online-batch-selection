import torch
import models
import timm

from models.BayesNet import CLIPZeroShotClassifier


def build_teacher_model(config, logger):
    teacher_config = config.get('diagnostics', {}).get('ntk_teacher_model', {})
    teacher_model_source = teacher_config.get('source', config.get('teacher_model_source'))
    teacher_model_path = teacher_config.get('path', config.get('teacher_model_path'))

    if teacher_model_source == 'clip':
        classes = config.get('classes', config.get('dataset', {}).get('classes'))
        template = config.get('template', config.get('dataset', {}).get('template'))
        if classes is None or template is None:
            raise ValueError('clip teacher models require classes and template in the config.')

        return CLIPZeroShotClassifier(
            classes,
            template,
            config['dataset']['name'],
            config['clip']['clip_architecture'],
            tau=config['clip']['tau'],
        )

    if teacher_model_source == 'timm':
        logger.info(f'Loading teacher model from {teacher_model_path}')
        return timm.create_model(teacher_model_path, pretrained=True)

    if teacher_model_source == 'local_pretrained':
        logger.info(f'Loading teacher model from {teacher_model_path}')
        lp_config = config.get('local_pretrained', {})
        if lp_config.get('type'):
            model_type = lp_config['type']
            model_args = (lp_config.get('params') or {}) | config['dataset']
        else:
            model_type = config['networks']['type']
            model_args = config['networks']['params'] | config['dataset']
        model = getattr(models, model_type)(**model_args)
        model.load_state_dict(torch.load(teacher_model_path, map_location='cpu'))
        return model

    raise ValueError(f'Teacher model type {teacher_model_source} not supported.')