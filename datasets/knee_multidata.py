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
        
        
        #dataset_list = ["fastmri_full", "oai_mri_full", "oai_xray_full", "sanyuan_full"]
        dataset_list = ["fastMRI_224_ex", "oai_mri_full", "oai_xray_full", "sanyuan_full", "sanyuan_xray_224", "sanyuan_xray_ex1_224"]
        self.split = kwargs.get("split")
        self.image_size = kwargs.get("size", 224)
        self.N = kwargs.get("N", None)
        assert self.split is not None
        self.file_list = []
        for dataset in dataset_list:
            dataset_path = os.path.join(root, dataset)
            for r, _, filenames in os.walk(dataset_path):
                folder = r.split('/')[-1]
                for filename in filenames:
                    label = 0
                    subfix = filename.split(".")[-1]
                    if subfix == "bmp" or subfix == "jpg" or subfix == "png":
                        pass
                    else:
                        continue
                    self.file_list.append([os.path.join(r, filename), label])
        dataset_append_list = ["patient_data5"]
        self.file_append_list = []
        for dataset in dataset_append_list:
            dataset_path = os.path.join(root, dataset)
            for r, _, filenames in os.walk(dataset_path):
                folder = r.split('/')[-1]
                for filename in filenames:
                    label = 0
                    subfix = filename.split(".")[-1]
                    if subfix == "bmp" or subfix == "jpg" or subfix == "png":
                        pass
                    else:
                        continue
                    self.file_append_list.append([os.path.join(r, filename), label])
        random.shuffle(self.file_append_list)
        self.file_list.extend(self.file_append_list[:400000])
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
    return knee_multi_dataset(**kwargs)
