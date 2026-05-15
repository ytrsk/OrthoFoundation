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



class omnimed_dataset(Dataset):

    def __init__(self, root, **kwargs):
        super(omnimed_dataset, self).__init__()
        root = os.path.join(root, "OmniMedVQA", "Images")
        self.split = kwargs.get("split")
        self.image_size = kwargs.get("size", 224)
        self.N = kwargs.get("N", None)
        assert self.split is not None
        self.file_list = []

        for r, _, filenames in os.walk(root):
            folder = r.split('/')[-1]
            for filename in filenames:
                label = 0
                subfix = filename.split(".")[-1]
                if subfix == "bmp" or subfix == "jpg" or subfix == "png":
                    pass
                else:
                    continue
                self.file_list.append([os.path.join(r, filename), label])
        print(len(self.file_list))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.trans_img = transforms.Compose([transforms.Normalize(mean, std)])

        random.seed(2333)
        random.shuffle(self.file_list)
        print(len(self.file_list))
        self.check_data()
        self.augmentation = DataAugmentationDINO()


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
    return omnimed_dataset(**kwargs)
