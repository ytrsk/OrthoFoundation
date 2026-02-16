import torch
from torch import nn
import ssl
import copy
import os
ssl._create_default_https_context = ssl._create_unverified_context

tag2tag = {
    "vitg16": "vitg16",
    "vitl16": "vitl16",
    "vitb16": "vitb16",
    "vits16": "vits16",
    "vitg16_reg": "vitg16_reg",
    "vitl16_reg": "vitl16_reg"
}

def get_vit(tag: str, repo_dir: str | None = None, weights: str | None = None):
    """
    Load DINOv3 ViT backbone.

    Default behavior follows the original backup code (local torch.hub repo + local weights),
    but allows overriding via kwargs/env to make the repo portable.
    """
    repo_dir = repo_dir or os.environ.get("DINOV3_REPO_DIR", "/root/autodl-tmp/dinov3")
    weights = weights or os.environ.get(
        "DINOV3_WEIGHTS",
        "/root/autodl-tmp/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    )
    # NOTE: hub model name depends on tag; keep original default (vitl16) for now.
    return torch.hub.load(repo_dir, "dinov3_vitl16", source="local", weights=weights)

class ViT(nn.Module):

    def __init__(self, **kwargs) -> None:
        super(ViT, self).__init__()
        self.backbone = get_vit(
            kwargs["backbone"],
            repo_dir=kwargs.get("repo_dir", None),
            weights=kwargs.get("weights", None),
        )
        self.mode = kwargs['mode']
        self.ft_nums = kwargs.get("ft_nums", len(self.backbone.blocks))
        self.ft_start = len(self.backbone.blocks) - self.ft_nums
        self.return_model_state = kwargs.get("return_model_state", False)
        if self.return_model_state == True:
            # for saving the orig_params during the first training infer
            self.orig_backbone = None

        for id, blk in enumerate(self.backbone.blocks):
            if id < self.ft_start:
                for name, parameter in blk.named_parameters():
                    parameter.requires_grad = False
        if self.mode in ['linear', 'prompt']:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad = False

        if self.mode == 'bias':
            for name, parameter in self.backbone.named_parameters():
                if 'bias' not in name:
                    parameter.requires_grad = False

        if self.mode == "prompt":
            self.n_prompt = kwargs.get("n_prompt", 1)
            n_blocks = len(self.backbone.blocks)
            self.prompt = nn.Parameter(torch.zeros(n_blocks, self.n_prompt,
                                                   self.backbone.embed_dim),
                                       requires_grad=True)
            nn.init.xavier_normal_(self.prompt.data)

    def forward_vit(self, x):
        feat = []
        b, c, h, w = x.shape
        if self.ft_start > 0:
            with torch.no_grad():
                x_at = self.backbone.prepare_tokens_with_masks(x)
        else:
            x_at = self.backbone.prepare_tokens_with_masks(x)

        x_at = x_at[0]
        if self.ft_start == 0:
            feat.append(x_at)

        if self.ft_start > 0:
            with torch.no_grad():
                for id, blk in enumerate(self.backbone.blocks[:self.ft_start]):
                    b, (c, m) = x_at.shape

                    x_at = blk(x_at)

                    if id == self.ft_start - 1:
                        x_at = x_at.clone().detach()
                        feat.append(x_at)

        for id, blk in enumerate(self.backbone.blocks[self.ft_start:]):
            if self.mode == "prompt":
                p = self.prompt[id]
                p = p.expand(b, self.n_prompt, m)
                x_at = torch.cat([p, x_at], 1)

            x_at = blk(x_at)

            if self.mode == "prompt":
                x_at = x_at[:, self.n_prompt:, :]

            if id == self.ft_start - 1:
                x_at = x_at.clone().detach()
                feat.append(x_at)

            # feat.append(x_at)
        x_at = self.backbone.norm(x_at)
        feat.append(x_at)
        return feat 
        # return x_at[:, :]

    def forward(self, x):
        if self.mode in ['linear']:
            with torch.no_grad():
                x = self.forward_vit(x)
        else:
            x = self.forward_vit(x)
        # x = self.head(x)
        if self.return_model_state:
            if self.orig_backbone == None:
                self.orig_backbone = copy.deepcopy(self.backbone)

            return x, self.backbone, self.orig_backbone
 
        return x


def Upsample(x, size):
    return nn.functional.interpolate(x,
                                     size=size,
                                     mode='bilinear',
                                     align_corners=False)



def get_model(**kwargs):
    return ViT(**kwargs)