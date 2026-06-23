import string
from datetime import datetime
import secrets
import os
import csv
import yaml

def get_configs(fname):
    with open(fname, 'r') as f:
        configs = yaml.safe_load(f)
        f.close()
    return configs

def random_str(num):
    salt = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(num))

    return salt

def get_date():

    now = datetime.now()
    return str(now.strftime("20%y_%h_%d_%H_%M_%S"))

def re_nest_configs(config_dict):
    flattened_params = [key for key in config_dict.keys() if '.' in key]
    for param in flattened_params:
        value = config_dict._items.pop(param)
        # value = config_dict[param]
        # del config_dict[param] 
        param_levels = param.split('.')
        parent = config_dict._items
        for level in param_levels:
            if isinstance(parent[level], dict):
                parent = parent[level]
            else:
                parent[level] = value

    if 'sweep_config' in config_dict.keys():
        config_dict._items.pop("sweep_config")

def get_save_dir(config, notes=None, exp_base='./exp/'):
    save_dir = exp_base
    save_dir = os.path.join(save_dir, config['dataset']['name'])
    save_dir = os.path.join(save_dir, config['method'])
    save_dir = save_dir + '_' + config['networks']['params']['m_type']
    save_dir = save_dir + '_bs' + str(config['training_opt']['batch_size'])
    save_dir = save_dir + '_ep' + str(config['training_opt']['num_epochs'])
    save_dir = save_dir + '_lr' + str(config['training_opt']['optim_params']['lr'])
    save_dir = save_dir + '_' + config['training_opt']['optimizer']
    save_dir = save_dir + '_' + config['training_opt']['scheduler']
    save_dir = save_dir + '_seed' + str(config['seed'])
    if 'method_opt' in config and 'ratio' in config['method_opt']:
        save_dir = save_dir + '_r' + str(config['method_opt']['ratio'])
    if notes is not None:
        save_dir = save_dir + '_' + notes
    return save_dir

class custom_logger():
    def __init__(self, output_path, name='log'):
        os.makedirs(output_path, exist_ok=True)
        now = datetime.now()
        logger_name = str(now.strftime("20%y_%h_%d_")) + name + ".txt"
        self.logger_path = os.path.join(output_path, logger_name)
        self.csv_path = os.path.join(output_path, logger_name.replace('.txt', '.csv'))
        # init logger file
        f = open(self.logger_path, "w+")
        f.write(self.get_local_time() + 'Start Logging \n')
        f.close()
        # init csv
        with open(self.csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([self.get_local_time(), ])

    def get_local_time(self):
        now = datetime.now()
        return str(now.strftime("%y_%h_%d %H:%M:%S : "))

    def info(self, log_str):
        print(str(log_str))
        with open(self.logger_path, "a") as f:
            f.write(self.get_local_time() + str(log_str) + '\n')

    def raise_error(self, error):
        prototype = '************* Error: {} *************'.format(str(error))
        self.info(prototype)
        raise ValueError(str(error))

    def info_iter(self, epoch, batch, total_batch, info_dict, print_iter):
        if batch % print_iter != 0:
            pass
        else:
            acc_log = 'Epoch {:5d}, Batch {:6d}/{},'.format(epoch, batch, total_batch)
            for key, val in info_dict.items():
                acc_log += ' {}: {:9.3f},'.format(str(key), float(val))
            self.info(acc_log)

    def write_results(self, result_list):
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(result_list)

    def wandb_init(self, config , project, name):
        # Imported here to keep utils' module-level imports light, because importing wandb takes a long time
        import wandb
        wandb.init(project=project, name=name, config=config)


    def wandb_log(self, log_dict, step=None):
        # Imported here to keep utils' module-level imports light, because importing wandb takes a long time
        import wandb
        if step is None:
            wandb.log(log_dict)
        else:
            wandb.log(log_dict, step=step)

    def wandb_finish(self):
        # Imported here to keep utils' module-level imports light, because importing wandb takes a long time
        import wandb
        wandb.finish()

