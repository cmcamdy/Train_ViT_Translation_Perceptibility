import torch.nn as nn
import torch


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Auxillary(nn.Module):
    def __init__(self,
                 sub_model,
                 dim,
                 out_cls=1,
                 with_scale=True,
                 with_trans=True):
        super().__init__()
        self.backbone = sub_model
        # self.mlp_head = nn.Sequential(nn.LayerNorm(dim*2), nn.Linear(dim*2, out_cls))
        self.with_trans = with_trans
        self.with_scale = with_scale
        if self.with_trans:
            # self.mlp_head_trans = nn.Sequential(nn.LayerNorm(dim*2), nn.Linear(dim*2, 2))
            self.mlp_head_trans = nn.Sequential(nn.LayerNorm(dim * 2),
                                                Mlp(dim * 2, 384, 2))
        if self.with_scale:
            # self.mlp_head_scale = nn.Sequential(nn.LayerNorm(dim*2), nn.Linear(dim*2, 1))
            self.mlp_head_scale = nn.Sequential(nn.LayerNorm(dim * 2),
                                                Mlp(dim * 2, 384, 2))

    def forward(self, x, x_translated):
        outs_x, latent_x = self.backbone(x)
        outs_x_translated, latent_x_translated = self.backbone(x_translated)
        # import pdb;pdb.set_trace()
        latent = torch.cat(
            (latent_x.mean(dim=1), latent_x_translated.mean(dim=1)), dim=1)
        # return outs_x, outs_x_translated, self.mlp_head_trans(latent), self.mlp_head(latent)
        if self.with_scale and self.with_trans:
            return outs_x, outs_x_translated, self.mlp_head_trans(
                latent), self.mlp_head_scale(latent)
        if self.with_scale and not self.with_trans:
            return outs_x, outs_x_translated, self.mlp_head_scale(latent)
        if not self.with_scale and self.with_trans:
            return outs_x, outs_x_translated, self.mlp_head_trans(latent)
