import sys
from typing import Optional

sys.path.append(".")
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.nn as nn
from torch import optim
from process.build import build_processes
from tools.utils import init_seeds, open_config, get_parameter_number, load_ckpt
from tools.to_log import to_log, set_file
import yaml
import numpy as np
from torch.backends import cudnn
import random
from torch import distributed as dist
from torch import autocast
import torch
import os
import argparse
import time

world_size = 1
local_rank = 0
device = None
def cosine_schedule(
    step: int,
    max_steps: int,
    start_value: float,
    end_value: float,
    period: Optional[int] = None
) -> float:
    """Gradually modify start_value to end_value using cosine decay.

    Args:
        step: Current step number.
        max_steps: Total number of steps.
        start_value: Starting value.
        end_value: Target value.
        period: Number of steps for a full cosine cycle, defaults to max_steps.

    Returns:
        Cosine decay value.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if period is not None and period <= 0:
        raise ValueError("period must be positive")
    
    # Use max_steps as period if period is None or enforce end_value at the last step
    effective_period = period if period is not None else max_steps
    
    # Special handling to avoid potential division by zero and ensure correct final value
    if step >= max_steps:
        return end_value
    
    decay = end_value + 0.5 * (start_value - end_value) * (1 + np.cos(np.pi * step / effective_period))
    return decay

def print_time(iter, stime, max_iter):
    t = time.time()-stime
    et = max_iter / iter * t - t
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    print("time: ", end="")
    print ("%02d:%02d:%02d" % (h, m, s), end="   ")
    print("eta: ", end="")
    m, s = divmod(et, 60)
    h, m = divmod(m, 60)
    print ("%02d:%02d:%02d" % (h, m, s))

def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    """
    Updates model_ema with Exponential Moving Average of model
    """
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1.0 - m)    
def train(args, root):
    to_log(args)
    args_train = args['train']
    seed = args_train.get('seed', 2106346)
    init_seeds(seed + local_rank)

    args_processes = args['processes']
    data_pool = {"opt": {}, "sch": {}}
    processes = build_processes(data_pool=data_pool,
                                local_rank=local_rank,
                                **args_processes)
    processes_collection = {x.process_name: x for x in processes}
    student_model = processes_collection["student_model"]
    teacher_model = processes_collection["teacher_model"]
    for p in teacher_model.model.parameters():
        p.requires_grad = False
    for process in processes:
        process.load(root=root, iter=args_train['load_iter'])
    torch.cuda.empty_cache()

    use_amp = args_train.get("amp", False)

    for _ in range(args_train['load_iter']):
        for process in processes:
            if process.sch is not None:
                process.sch.step()

    writer = SummaryWriter(os.path.join(root, "logs/result/event/"))

    stime = time.time()
    init_iter = data_pool['tot_iter']
    for iter in range(data_pool['tot_iter'] + 1, args_train['max_iter'] + 1):
        data_pool['state'] = "train"
        data_pool['tot_iter'] = iter
        for process in processes:
            process.run()
        momentum = cosine_schedule(data_pool['tot_iter'], args_train['max_iter'], 0.996, 1)
        update_momentum(student_model.model, teacher_model.model, momentum)
        for name, opt in data_pool['opt'].items():
            opt.zero_grad()
        tot_loss = 0.

        with autocast(device_type="cuda", enabled=use_amp):
            for name, loss in data_pool['loss'].items():
                tot_loss += loss

        tot_loss.backward()
        for name, opt in data_pool['opt'].items():
            opt.step()
        for name, sch in data_pool['sch'].items():
            sch.step()

        if data_pool['tot_iter'] % args_train[
                'show_interval'] == 0 and local_rank == 0:
            to_log("iter: {}, epoch: {}, batch: {}/{}, loss: {}".format(
                data_pool['tot_iter'], data_pool['epoch'], data_pool['batch'],
                data_pool['train_batch'], tot_loss))
            writer.add_scalar("train/loss", tot_loss, data_pool['tot_iter'])

            for name, loss in data_pool['loss'].items():
                writer.add_scalar("train/loss_{}".format(name), loss,
                                    data_pool['tot_iter'])

            for name, opt in data_pool['opt'].items():
                writer.add_scalar("train/lr_{}".format(name),
                                  opt.param_groups[0]['lr'],
                                  data_pool['tot_iter'])

        if data_pool['tot_iter'] % args_train['snapshot_interval'] == 0:
            for process in processes:
                process.save(root=root)

        if data_pool['tot_iter'] % args_train[
                'valid_interval'] == 0 and local_rank == 0:
            with torch.no_grad():
                data_pool['state'] = "valid"
                #print(data_pool['valid_batch'])
                for val_iter in range(data_pool['valid_batch']):
                    for process in processes:
                        #print(process.process_name)
                        process.run()

            tot_loss = 0.
            for name, loss in data_pool['loss'].items():
                tot_loss += loss
                to_log("Loss: {}: {:.5f}".format(name, loss))
                writer.add_scalar("valid/loss/{}".format(name),
                                  loss, data_pool['tot_iter'])
            for name, loss in data_pool.get('metric', {}).items():
                to_log("Metric: {}: {:.5f}".format(name, loss))
                writer.add_scalar("valid/{}".format(name),
                                  loss, data_pool['tot_iter'])
            writer.add_scalar("valid/loss", tot_loss, data_pool['tot_iter'])

            for name, info in data_pool.get('addition', {}).items():
                to_log("addition info: {}: {}".format(name, str(info)))
            #print(root)
            print_time(iter - init_iter, stime, args_train['max_iter'] - init_iter)

    writer.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--gpus", type=str)
    args=parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
    # torch.multiprocessing.set_start_method('spawn')

    # Support both:
    # - torchrun / distributed launch (LOCAL_RANK exists)
    # - plain single-process run (no LOCAL_RANK)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        assert torch.cuda.device_count() > local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if "LOCAL_RANK" in os.environ:
        import datetime
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=datetime.timedelta(seconds=36000),
        )
        world_size = dist.get_world_size()
    else:
        world_size = 1

    if local_rank == 0:
        if not os.path.exists(os.path.join(args.root, "logs/result/event")):
            os.makedirs(os.path.join(args.root, "logs/result/event"))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    with open(os.path.join(args.root, "logs/log.txt"), "a+") as log_file:
        log_file.truncate(0)
        set_file(log_file, rank=local_rank)
    
        train(open_config(args.root), args.root)
