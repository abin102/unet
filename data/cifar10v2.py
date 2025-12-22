import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as datasets

import numpy as np
from PIL import Image
import random
import math
import os
# from .autoaugment import CIFAR10Policy



class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root=None, imb_type='exp', imb_factor=0.01, rand_number=0,
                 train=True, transform=None, target_transform=None,
                 download=True, class_balance=False):

        if root is None:
            root = os.path.expanduser("./data")   # ✅ user-writable default

        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

        self.class_balance = class_balance
        if class_balance:
            self.class_data = [[] for _ in range(self.cls_num)]
            for i in range(len(self.targets)):
                self.class_data[self.targets[i]].append(i)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
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
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        return [self.num_per_cls_dict[i] for i in range(self.cls_num)]

    def __getitem__(self, index):
        if self.class_balance:
            sample_class = random.randint(0, self.cls_num - 1)
            index = random.choice(self.class_data[sample_class])
            img, target = self.data[index], sample_class
        else:
            img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class CIFAR10V2(object):
    def __init__(self, batch_size=128, class_balance=False, imb_factor=None, 
                 imb_type='exp'):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        normalize = transforms.Normalize(mean, std)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        root = os.path.expanduser("./data")
        trainset = IMBALANCECIFAR10(root=root, train=True, transform=transform_train,
                                    download=True, imb_factor=imb_factor,
                                    class_balance=class_balance)
        testset = datasets.CIFAR10(root=root, train=False, transform=transform_test, download=True)

        self.train = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=True, drop_last=True)

        self.test = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True) 


