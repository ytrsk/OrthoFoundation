import random
import numpy as np
import torch
from torch.backends import cudnn
import os
import yaml
from tools.to_log import to_log


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False


def open_config(root, name="config.yaml"):
    f = open(os.path.join(root, name))
    config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load_ckpt(models, epoch, root):

    def _detect_latest():
        for name in models.keys():
            checkpoints = os.listdir(os.path.join(root, "logs"))
            checkpoints = [
                f for f in checkpoints
                if f.startswith("{}_epoch_".format(name)) and f.endswith(".pth")
            ]
            checkpoints = [
                int(f[len("{}_epoch_".format(name)):-len(".pth")]) for f in checkpoints
            ]
            checkpoints = sorted(checkpoints)
            _epoch = checkpoints[-1] if len(checkpoints) > 0 else None
            return _epoch

    if epoch == -1:
        epoch = _detect_latest()
    if epoch is None:
        return -1
    for name, model in models.items():
        pth_path = os.path.join(root,
                                "logs/" + name + "_epoch_{}.pth".format(epoch))
        if not os.path.exists(pth_path):
            to_log("can't find pth file: {}".format(name))
            continue
        state_dict = model.state_dict()
        ckpt = torch.load(pth_path, map_location="cpu")
        load_ckpt = {k: v for k, v in ckpt.items() if k in state_dict.keys()}
        print(len(load_ckpt))
        state_dict.update(load_ckpt)
        load_ckpt = {k[7:]: v for k, v in ckpt.items() if k[7:]
                     in state_dict.keys()}
        print(len(load_ckpt))
        state_dict.update(load_ckpt)
        model.load_state_dict(state_dict, strict=True)
        to_log("load model: {} from iter: {}".format(name, epoch))
    return epoch

def load_pre_trained_ckpt(models, pth_path):
    for name, model in models.items():
        if not os.path.exists(pth_path):
            to_log("can't find pth file: {}".format(name))
            assert False
            continue
        state_dict = model.state_dict()
        ckpt = torch.load(pth_path, map_location="cpu")
        # for k, v in ckpt.items():
        #     if k not in state_dict.keys():
        #         print(k)
        # assert False
        load_ckpt = {k: v for k, v in ckpt.items() if k in state_dict.keys()}
        
        #print(ckpt)
        #assert False
        state_dict.update(load_ckpt)
        load_ckpt = {k[7:]: v for k, v in ckpt.items() if k[7:]
                     in state_dict.keys()}
        print(len(load_ckpt))
        state_dict.update(load_ckpt)
        #print(state_dict)
        model.load_state_dict(state_dict, strict=True)
        to_log("load model from" + str(pth_path))
    return -1