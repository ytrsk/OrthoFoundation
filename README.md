## OrthoFoundation

预训练代码（DINOv2 / DINOv3）。

- 入口：`tools/train.py`
- 配置：`exp/pretrain/*/config.yaml`

### 跑起来

单卡：

```bash
python tools/train.py --root exp/pretrain/dinov3/39w --gpus 0
```

多卡（按环境调整 `--nproc_per_node`）：

```bash
torchrun --nproc_per_node=1 tools/train.py --root exp/pretrain/dinov3/39w --gpus 0
```

数据路径在 `config.yaml` 里改：`processes.*.args.root`


