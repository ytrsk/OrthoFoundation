import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import torch
import random
import math
import matplotlib as plt
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from datasets.augmentation import DataAugmentationDINO



class knee_multi_dataset(Dataset):

    def __init__(self, root, **kwargs):
        super(knee_multi_dataset, self).__init__()
        
        self.split = kwargs.get("split")
        self.image_size = kwargs.get("size", 224)
        self.N = kwargs.get("N", None)
        assert self.split is not None
        self.file_list = []

        dataset_dirs = kwargs.get("dataset_dirs", None)
        dataset_paths = [root] if dataset_dirs is None else [
            os.path.join(root, dataset_dir) for dataset_dir in dataset_dirs
        ]
        for dataset_path in dataset_paths:
            self.file_list.extend(self.scan_images(dataset_path))

        dataset_append_list = kwargs.get("append_dirs", [])
        append_limit = kwargs.get("append_limit", None)
        self.file_append_list = []
        for dataset in dataset_append_list:
            dataset_path = os.path.join(root, dataset)
            self.file_append_list.extend(self.scan_images(dataset_path))
        random.shuffle(self.file_append_list)
        if append_limit is not None:
            self.file_append_list = self.file_append_list[:append_limit]
        self.file_list.extend(self.file_append_list)
        print(len(self.file_list))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.trans_img = transforms.Compose([transforms.Normalize(mean, std)])

        random.seed(2333)
        random.shuffle(self.file_list)
        print(len(self.file_list))
        self.check_data()
        self.augmentation = DataAugmentationDINO()

    def scan_images(self, dataset_path):
        file_list = []
        for r, _, filenames in os.walk(dataset_path):
            for filename in filenames:
                subfix = filename.split(".")[-1].lower()
                if subfix not in ["bmp", "jpg", "jpeg", "png"]:
                    continue
                file_list.append([os.path.join(r, filename), 0])
        return file_list

    def check_data(self):
        for filepath, label in tqdm(self.file_list):
            assert os.path.exists(filepath)

    def __len__(self):
        if self.N:
            return self.N
        return len(self.file_list)

    def __getitem__(self, id):
        if self.N:
            id = id % self.N
        filepath, label = self.file_list[id]
        img = Image.open(filepath).convert("RGB")
        img = self.augmentation(img)
        return img


def get_dataset(**kwargs):
    return knee_multi_dataset(**kwargs)
