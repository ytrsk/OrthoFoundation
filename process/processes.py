from __future__ import annotations

import os
import random
import sys
from typing import Dict

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from datasets import knee_multidata, omnimed
from metric import get_metric
from nets import dinov2_vit, dinov3_vit
from nets.dino_head import DINOHead
from nets.multicrop_wrapper import MultiCropWrapper
from tools.to_log import to_log
from tools.utils import get_parameter_number, load_pre_trained_ckpt


DATASET_REGISTRY: Dict[str, object] = {
    "knee_multidata": knee_multidata,
    "omnimed": omnimed,
}

MODEL_REGISTRY: Dict[str, object] = {
    "dinov2_vit": dinov2_vit,
    "dinov3_vit": dinov3_vit,
}


class base_process():

    def __init__(self, data_pool, local_rank, workplace, process_name) -> None:
        self.data_pool = data_pool
        self.local_rank = local_rank
        if torch.cuda.is_available():
            self.device = torch.device('cuda', local_rank)
            self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        else:
            self.device = torch.device("cpu")
            self.world_size = 1
        self.input_name = []
        self.output_name = []
        self.lst_state = None
        self.state = None
        self.workplace = workplace
        self.process_name = process_name

    def is_in_workplace(self, state: str):
        if len(self.workplace) > 0 and state not in self.workplace:
            return False
        return True

    def run(self, **kwargs):
        self.update_state()
        if not self.is_in_workplace(self.state):
            return
        self.put_data(self.infer())

    def infer(self, **kwargs):
        pass

    def load(self, **kwargs):
        pass

    def save(self, **kwargs):
        pass

    def get_data(self):
        ret = []
        for name in self.input_name:
            ret.append(self.data_pool[name])
        return ret

    def put_data(self, *_data):
        # print(self.output_name)
        # print(len(*_data))
        while len(self.output_name) != len(_data):
            _data = _data[0]
        for id, name in enumerate(self.output_name):
            names = name.split('/')
            now = self.data_pool
            for d in names[:-1]:
                if d not in now:
                    now[d] = {}
                now = now[d]
            try:
                _d = _data[id].to(self.device)
            except:
                _d = _data[id]
            now[names[-1]] = _d

    def update_state(self):
        self.lst_state = self.state
        self.state = self.data_pool['state']


class UnmixSampler(Sampler):
    def __init__(self, data_source, bs, shuffle=True, rank=0):
        self.data_source = data_source
        self.datanums = self.data_source.datanums
        self.bs = bs
        self.epoch = 0
        self.shuffle = shuffle
        random.seed(2333 + rank)
        indices = []

        for i in range(1, len(self.datanums)):
            self.datanums[i] += self.datanums[i-1]
        self.datanums = [0] + self.datanums
        for i in range(1, len(self.datanums)):
            nums = self.datanums[i] - self.datanums[i-1]
            indices_now = []
            indices_now = list(range(self.datanums[i-1], self.datanums[i]))
            random.shuffle(indices_now)
            pad_num = self.bs-nums % self.bs
            indices_now += indices_now[:pad_num]
            indices += indices_now
        self.indices = [indices[i:i+self.bs]
                        for i in range(0, len(indices), self.bs)]

    def __iter__(self):
        indices = self.indices
        if self.shuffle:
            random.shuffle(indices)
        indices = [x for batch in indices for x in batch]
        return iter(indices)

    def __len__(self):
        return len(self.indices * self.bs)

    def set_epoch(self, epoch):
        self.epoch = epoch


class data_process(base_process):

    def __init__(self, data_pool, local_rank, **kwargs) -> None:
        super(data_process, self).__init__(data_pool, local_rank,
                                           kwargs.get('workplace', []),kwargs.get('process_name', ""))
        # self.input_name = kwargs['input_name']
        self.output_name = kwargs['output_name']
        tag = kwargs["tag"]
        if tag not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset tag: {tag}. Available: {sorted(DATASET_REGISTRY.keys())}")
        dataset = DATASET_REGISTRY[tag]
        self.train_dataset = dataset.get_dataset(**kwargs['args'],
                                                 split="train")
        self.valid_dataset = dataset.get_dataset(**kwargs['args'],
                                                 split="valid")
        self.num_workers = kwargs['num_workers']
        self.bs = kwargs['bs'] // self.world_size
        self.bs_val = kwargs.get('bs_val', 1)
        self.unmix = kwargs.get("unmix", False)

        self.train_loader, self.sampler = self.get_loader("train")
        self.valid_loarder, _ = self.get_loader("valid")
        if "valid_batch" not in self.data_pool:
            self.data_pool["valid_batch"] = 0
        if "train_batch" not in self.data_pool:
            self.data_pool["train_batch"] = 0
        if self.is_in_workplace("train"):
            self.data_pool['train_batch'] += len(self.train_loader)
        if self.is_in_workplace("valid"):
            self.data_pool['valid_batch'] = max(
                [len(self.valid_loarder), self.data_pool['valid_batch']])
        self.iter_train = iter(self.train_loader)
        self.iter_valid = iter(self.valid_loarder)
        # print(len(self.train_dataset), self.data_pool['train_batch'])

    def get_loader(self, split):
        if split == "train":
            if not self.unmix:
                if dist.is_available() and dist.is_initialized() and self.world_size > 1:
                    sampler = DistributedSampler(
                        self.train_dataset,
                        num_replicas=self.world_size,
                        rank=self.local_rank,
                        shuffle=True,
                    )
                    shuffle = False
                else:
                    sampler = None
                    shuffle = True
            else:
                sampler = UnmixSampler(
                    self.train_dataset,
                    bs=self.bs,
                    rank=self.local_rank,
                )
                shuffle = False
            bs = self.bs
            nw = self.num_workers
            dataset = self.train_dataset

        else:
            bs = self.bs_val
            sampler = None
            if self.unmix:
                sampler = UnmixSampler(
                    self.valid_dataset,
                    bs=self.bs_val,
                    rank=self.local_rank,
                    shuffle=False
                )
            nw = min(self.num_workers, self.bs_val)
            dataset = self.valid_dataset
        dataloader = DataLoader(dataset,
                                bs,
                                shuffle=shuffle if split == "train" else False,
                                num_workers=nw,
                                sampler=sampler)
        return dataloader, sampler

    def load(self, root, iter):
        self.data_pool['tot_iter'] = max(iter, 0)

    def infer(self):
        if self.state == "train":
            epoch = (self.data_pool['tot_iter'] -
                     1) // self.data_pool['train_batch']
            # if self.data_pool['tot_iter'] % self.data_pool['train_batch'] == 1:
            # if self.data_pool['tot_iter'] - epoch * self.data_pool['train_batch'] == 1:
            #     self.sampler.set_epoch(epoch)
            #     self.iter_train = iter(self.train_loader)
            self.data_pool['epoch'] = epoch
            self.data_pool['batch'] = (self.data_pool['tot_iter'] -
                                       1) % self.data_pool['train_batch'] + 1
            self.iter_data = self.iter_train
        elif self.state != self.lst_state:
            self.iter_valid = iter(self.valid_loarder)
            self.iter_data = self.iter_valid

        try:
            x = next(self.iter_data)
        except:
            if self.state == "train":
                epoch = (self.data_pool['tot_iter'] -
                         1) // len(self.train_loader)
                self.sampler.set_epoch(epoch)
                self.iter_train = iter(self.train_loader)
                self.iter_data = self.iter_train
            else:
                self.iter_valid = iter(self.valid_loarder)
                self.iter_data = self.iter_valid
            # print("data reset")
            x = next(self.iter_data)

        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        elif isinstance(x, list):
            for v in x:
                if isinstance(v, torch.Tensor):
                    v.to(self.device)

        elif isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    v.to(self.device)
        return x


class model_process(base_process):

    def __init__(self, data_pool, local_rank, **kwargs) -> None:
        super(model_process, self).__init__(data_pool, local_rank,
                                            kwargs.get('workplace', []),kwargs.get('process_name', ""))
        self.input_name = kwargs['input_name']
        self.output_name = kwargs['output_name']
        self.model_name = kwargs['tag']
        self.training = kwargs.get("training", True)
        self.pth_path = kwargs.get("pth_path", None)
        self.load_iter = kwargs.get("load_iter", -1)
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model tag: {self.model_name}. Available: {sorted(MODEL_REGISTRY.keys())}")
        self.model = MODEL_REGISTRY[self.model_name]
        self.model = self.model.get_model(**kwargs.get("args", {}))
    
        if "crop_wrapper" in kwargs.keys(): 
            self.model = MultiCropWrapper(self.model, DINOHead(), kwargs["crop_wrapper"])
        self.wrapper=True
        self.model = self.model.to(self.device)
        self.keep_eval = False

        if kwargs.get('ddp', False) and torch.cuda.is_available() and dist.is_available() and dist.is_initialized() and self.world_size > 1:
            self.model = DDP(self.model,
                             device_ids=[self.local_rank],
                             find_unused_parameters=True)
        to_log(get_parameter_number(self.model))
        if "opt" in kwargs:
            self.build_opt(**kwargs['opt'])
            self.data_pool['opt'][self.model_name] = self.opt
        if "sch" in kwargs:
            self.build_sch(**kwargs['sch'])
            self.data_pool['sch'][self.model_name] = self.sch

    def build_opt(self, tag, **args):
        self.opt = getattr(optim, tag)(params=self.model.parameters(), **args)

    def build_sch(self, tag, **args):
        self.sch = getattr(optim.lr_scheduler, tag)(optimizer=self.opt, **args)

    def load(self, root, iter):
        if self.pth_path is not None:
            iter = load_pre_trained_ckpt({
                self.model_name: self.model,
            }, self.pth_path)
        # if self.training:
        #     iter = load_ckpt({
        #         self.model_name: self.model,
        #     }, iter, root)
        self.data_pool["tot_iter"] = max(iter, 0)

    def save(self, root):
        if self.training:
            path = os.path.join(
                root, "logs/{}_epoch_{}.pth".format(self.model_name,
                                                    self.data_pool['tot_iter']))
            if self.wrapper:
                torch.save(self.model.module.backbone.state_dict(), path.replace(self.model_name, self.model_name+"backbone"))
            torch.save(self.model.state_dict(), path)

    def eval(self):
        self.model.eval()
        self.keep_eval = True

    def infer(self):
        if self.state != self.lst_state:
            if self.state == "train" and not self.keep_eval:
                self.model.train()
            elif self.state != "train":
                self.model.eval()

        if not self.training:
            with torch.no_grad():
                x = self.get_data()
                if len(x) == 1:
                    y = self.model(*x)
                else:
                    y = self.model(x)
                return y
        else:
            x = self.get_data()
            if len(x) == 1:
                y = self.model(*x)
            else:
                y = self.model(*x)

            return y


class metric_process(base_process):

    def __init__(self, data_pool, local_rank, **kwargs) -> None:
        super(metric_process, self).__init__(data_pool, local_rank,
                                             kwargs.get('workplace', []),kwargs.get('process_name', ""))
        self.input_name = kwargs['input_name']
        self.output_name = kwargs['output_name']
        self.model_name = kwargs['tag']
        self.model = get_metric(self.model_name, **kwargs.get("args", {}))
        self.model = self.model.to(self.device)
        self.ratio = kwargs.get('lambda', 1.0)

    def infer(self):
        if self.state == "train":
            self.model.reset()
        if self.state != self.lst_state:
            if self.state != "train":
                self.model.reset()

        x = self.get_data()

        self.model(*x)
        y = self.model.get()
        y = y * self.ratio
        if hasattr(self.model, "get_addition"):
            addition = self.model.get_addition()
        else:
            addition = None
        if addition is not None:
            return y, addition
        return y
