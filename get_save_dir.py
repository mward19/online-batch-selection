import argparse
from utils import get_save_dir, get_configs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--optim', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--notes', type=str,
                        default=None, 
                        help='Notes for the experiment.')
    args = parser.parse_args()

    method_config = get_configs(args.method)
    data_config = get_configs(args.data)
    model_config = get_configs(args.model)
    optim_config = get_configs(args.optim)
    config = {**method_config, **data_config, **model_config, **optim_config}
    config['seed'] = args.seed
    
    print(get_save_dir(config, args.notes))

    