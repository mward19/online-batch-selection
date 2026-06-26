from torchvision import datasets, transforms
import torch
import numpy as np
import random

from .data_utils.generate_noise import apply_or_generate_label_noise

cifar3_classes = ['dog','cat','bird',]
cifar3_templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

cifar10_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck',]
cifar10_templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

cifar100_classes = [
    'apple',
    'aquarium fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'keyboard',
    'lamp',
    'lawn mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak tree',
    'orange',
    'orchid',
    'otter',
    'palm tree',
    'pear',
    'pickup truck',
    'pine tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow tree',
    'wolf',
    'woman',
    'worm',
]

cifar100_templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

class wrapped_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # Some dataset wrappers (e.g., Subset) may not expose .targets
        self.targets = getattr(dataset, 'targets', None)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return {
            'input': self.dataset[index][0],
            'target': self.dataset[index][1],
            'index': index
        }

class IMBALANCECIFAR10(datasets.CIFAR10):
    # From: https://github.com/kaidic/LDAM-DRW
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False, reverse=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, reverse)
        self.gen_imbalanced_data(img_num_list)
        self.reverse = reverse

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num =  img_max * (imb_factor**((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))                    
                else:
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    # From: https://github.com/kaidic/LDAM-DRW
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


def _build_test_loader(config, dst_test):
    config['training_opt']['test_batch_size'] = config['training_opt']['batch_size'] if 'test_batch_size' not in config['training_opt'] else config['training_opt']['test_batch_size']
    return torch.utils.data.DataLoader(
        wrapped_dataset(dst_test), batch_size = config['training_opt']['test_batch_size'],
        shuffle=False, num_workers = config['training_opt']['num_data_workers'], pin_memory=True, drop_last=False
    )


def _build_dataset_info(config, logger, dataset_name, dst_train, dst_test, num_classes, classes, templates, include_noise=False):
    payload = {
        'num_classes': num_classes,
        'train_dset': wrapped_dataset(dst_train),
        'test_loader': _build_test_loader(config, dst_test),
        'num_train_samples': len(dst_train),
        'classes': classes,
        'template': templates,
    }
    if include_noise:
        payload.update(
            apply_or_generate_label_noise(
                dataset=dst_train,
                num_classes=num_classes,
                dataset_config=config['dataset'],
                logger=logger,
                dataset_name=dataset_name,
                seed=config.get('seed'),
                run_dir=config.get('save_dir'),
            )
        )
        payload['train_dset'] = wrapped_dataset(dst_train)
    return payload

def CIFAR3(config, logger):
    im_size = (32, 32) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 3
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 32 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 32 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )
    
    dst_train = datasets.CIFAR10(
        config['dataset']['root'], train=True, download=True, transform= transform
    )
    
    
    dst_test = datasets.CIFAR10(config['dataset']['root'], train=False, download=True, transform = test_transform)
    
    # Keep 3 CIFAR-10 classes: dog(5), cat(3), bird(2)
    keep = [5, 3, 2]

    # Filter train split in-place so we preserve CIFAR dataset fields (.data/.targets)
    train_targets = np.array(dst_train.targets, dtype=np.int64)
    train_mask = np.isin(train_targets, keep)
    dst_train.data = dst_train.data[train_mask]
    # Remap labels to contiguous [0, 1, 2] for safety
    label_map = {old: new for new, old in enumerate(keep)}
    dst_train.targets = [label_map[int(t)] for t in train_targets[train_mask]]

    # Filter test split in-place with the same mapping
    test_targets = np.array(dst_test.targets, dtype=np.int64)
    test_mask = np.isin(test_targets, keep)
    dst_test.data = dst_test.data[test_mask]
    dst_test.targets = [label_map[int(t)] for t in test_targets[test_mask]]

    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='CIFAR3',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=cifar3_classes,
        templates=cifar3_templates,
    )


def CIFAR3_Noise(config, logger):
    im_size = (32, 32) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 3
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 32 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 32 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    dst_train = datasets.CIFAR10(
        config['dataset']['root'], train=True, download=True, transform= transform
    )

    dst_test = datasets.CIFAR10(config['dataset']['root'], train=False, download=True, transform = test_transform)

    keep = [5, 3, 2]
    train_targets = np.array(dst_train.targets, dtype=np.int64)
    train_mask = np.isin(train_targets, keep)
    dst_train.data = dst_train.data[train_mask]
    label_map = {old: new for new, old in enumerate(keep)}
    dst_train.targets = [label_map[int(t)] for t in train_targets[train_mask]]

    test_targets = np.array(dst_test.targets, dtype=np.int64)
    test_mask = np.isin(test_targets, keep)
    dst_test.data = dst_test.data[test_mask]
    dst_test.targets = [label_map[int(t)] for t in test_targets[test_mask]]

    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='CIFAR3',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=cifar3_classes,
        templates=cifar3_templates,
        include_noise=True,
    )

def CIFAR10(config, logger):
    im_size = (32, 32) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 32 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 32 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )
    
    dst_train = datasets.CIFAR10(
        config['dataset']['root'], train=True, download=True, transform= transform
    )
    
    
    dst_test = datasets.CIFAR10(config['dataset']['root'], train=False, download=True, transform = test_transform)
    # class_names = dst_train.classes
    # dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
    # dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)
    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='CIFAR10',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=cifar10_classes,
        templates=cifar10_templates,
    )

def CIFAR10_minimal(config, logger):
    """Gets a subset of CIFAR10 for debugging purposes."""
    im_size = (32, 32) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    dst_train_all = datasets.CIFAR10(config['dataset']['root'], train=True, download=True, transform=transform)
    dst_test_all = datasets.CIFAR10(config['dataset']['root'], train=False, download=True, transform=transform)
    
    # Only keep a portion of the training and testing data
    keep_ratio_train = config['dataset']['keep_ratio_train']
    keep_ratio_test = config['dataset']['keep_ratio_test']

    indices_train = torch.randperm(len(dst_train_all))[:int(len(dst_train_all) * keep_ratio_train)]
    dst_train = torch.utils.data.Subset(dst_train_all, indices_train)

    indices_test = torch.randperm(len(dst_test_all))[:int(len(dst_test_all) * keep_ratio_test)]
    dst_test = torch.utils.data.Subset(dst_test_all, indices_test)

    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='CIFAR10',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=cifar10_classes,
        templates=cifar10_templates,
    )

def CIFAR10_Noise(config, logger):
    im_size = (32, 32) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 32 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 32 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )
    
    dst_train = datasets.CIFAR10(
        config['dataset']['root'], train=True, download=True, transform= transform
    )
    
    
    dst_test = datasets.CIFAR10(config['dataset']['root'], train=False, download=True, transform = test_transform)
    # class_names = dst_train.classes
    # dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
    # dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)
    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='CIFAR10',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=cifar10_classes,
        templates=cifar10_templates,
        include_noise=True,
    )
    
    
def CIFAR10_LT(config, logger):
    im_size = (32, 32) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 32 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 32 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )
    
    # dst_train = datasets.CIFAR10(
    #     config['dataset']['root'], train=True, download=True, transform= transform
    # )
    dst_train = IMBALANCECIFAR10(root = config['dataset']['root'], imb_factor = config['dataset']['imb_factor'], rand_number = config['dataset']['rand_number'], train = True, download= True, transform= transform)
    dst_test = datasets.CIFAR10(config['dataset']['root'], train=False, download=True, transform = test_transform)
    # class_names = dst_train.classes
    # dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
    # dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)
    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='CIFAR10_LT',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=cifar10_classes,
        templates=cifar10_templates,
    )
  
def CIFAR100(config, logger):
    im_size = (32, 32) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    print(f'Image size: {im_size}')
    num_classes = 100
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 32 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )
    
    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 32 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )
    
    dst_train = datasets.CIFAR100(
        config['dataset']['root'], train=True, download=True, transform= transform
    )
    
    dst_test = datasets.CIFAR100(config['dataset']['root'], train=False, download=True, transform = test_transform)
    # class_names = dst_train.classes
    dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
    dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)
    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='CIFAR100',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=cifar100_classes,
        templates=cifar100_templates,
    )


def CIFAR100_Noise(config, logger):
    im_size = (32, 32) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    print(f'Image size: {im_size}')
    num_classes = 100
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 32 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 32 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    dst_train = datasets.CIFAR100(
        config['dataset']['root'], train=True, download=True, transform= transform
    )

    dst_test = datasets.CIFAR100(config['dataset']['root'], train=False, download=True, transform = test_transform)
    dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
    dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)

    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='CIFAR100',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=cifar100_classes,
        templates=cifar100_templates,
        include_noise=True,
    )

def CIFAR100_LT(config, logger):
    im_size = (32, 32) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    print(f'Image size: {im_size}')
    num_classes = 100
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 32 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )
    
    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 32 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )
    
    # dst_train = datasets.CIFAR100(
    #     config['dataset']['root'], train=True, download=True, transform= transform
    # )
    dst_train = IMBALANCECIFAR100(root = config['dataset']['root'], imb_factor = config['dataset']['imb_factor'], rand_number = config['dataset']['rand_number'], train = True, download= True, transform= transform)
    dst_test = datasets.CIFAR100(config['dataset']['root'], train=False, download=True, transform = test_transform)
    # class_names = dst_train.classes
    dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
    dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)
    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='CIFAR100_LT',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=cifar100_classes,
        templates=cifar100_templates,
    )