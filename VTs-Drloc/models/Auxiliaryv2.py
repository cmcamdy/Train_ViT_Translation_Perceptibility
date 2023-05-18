
import torch.nn as nn
import torch


class Auxillary(nn.Module):
    def __init__(self, sub_model, dim, out_cls=1, with_scale=True, with_trans=True):
        super().__init__()
        self.backbone = sub_model
        # self.mlp_head = nn.Sequential(nn.LayerNorm(dim*2), nn.Linear(dim*2, out_cls))
        self.with_trans = with_trans
        self.with_scale = with_scale
        if self.with_trans:
            self.mlp_head_trans = nn.Sequential(nn.LayerNorm(dim*2), nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, 2))
        if self.with_scale:
            self.mlp_head_scale = nn.Sequential(nn.LayerNorm(dim*2), nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, x, x_translated):
        outs_x = self.backbone(x)
        outs_x_translated = self.backbone(x_translated)
        latent = torch.cat((outs_x.latent, outs_x_translated.latent), dim=1)
        # return outs_x, outs_x_translated, self.mlp_head_trans(latent), self.mlp_head(latent)
        if self.with_scale and self.with_trans:
            return outs_x, outs_x_translated, self.mlp_head_trans(latent), self.mlp_head_scale(latent)
        if self.with_scale and not self.with_trans:
            return outs_x, outs_x_translated, self.mlp_head_scale(latent)
        if not self.with_scale and self.with_trans:
            return outs_x, outs_x_translated, self.mlp_head_trans(latent)
