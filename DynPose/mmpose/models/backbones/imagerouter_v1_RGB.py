# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmengine.model import BaseModule, constant_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
import torch.nn.functional as F
from mmpose.registry import MODELS
from .base_backbone import BaseBackbone


@MODELS.register_module()
class ImageRouter(BaseBackbone):

    def __init__(self):
        # Protect mutable default arguments
        super(ImageRouter, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.router = AdaptiveRouter([512],3)

        norm_cfg = dict(type='BN')
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, 96, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, 256, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, 256, postfix=3)
        self.norm4_name, norm4 = build_norm_layer(
            norm_cfg, 512, postfix=4)


        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)
        self.add_module(self.norm3_name, norm3)
        self.add_module(self.norm4_name, norm4)

        self.relu = nn.ReLU(inplace=True)


    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm3_name)

    @property
    def norm4(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm4_name)

    def forward(self, x):
        """Forward function."""

        # print("I'm in ImageRouter")
        outs = []
        outs.append(x)
        x = self.avg_pool(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu(x)

        # print("x",x.shape)
        scores = self.router([x])
        # print("scores",scores.shape)

        outs.append(scores)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        # self._freeze_stages()
        # if mode and self.norm_eval:
        #     for m in self.modules():
        #         # trick: eval have effect on BatchNorm only
        #         if isinstance(m, _BatchNorm):
        #             m.eval()

class AdaptiveRouter(nn.Module):
    def __init__(self, features_channels, out_channels, reduction=4):
        super(AdaptiveRouter, self).__init__()
        self.inp = sum(features_channels)
        self.oup = out_channels
        self.reduction = reduction
        self.layer1 = nn.Conv2d(self.inp, self.inp // self.reduction, kernel_size=1, bias=True)
        self.layer2 = nn.Conv2d(self.inp // self.reduction, self.oup, kernel_size=1, bias=True)

    def forward(self, xs):
        # xs = [x.mean(dim=(2, 3), keepdim=True) for x in xs] # rtmo
        xs = [x[:,:,8,6].unsqueeze(-1).unsqueeze(-1) for x in xs] # sig
        xs = torch.cat(xs, dim=1)
        xs = self.layer1(xs)
        xs = F.relu(xs, inplace=True)
        xs = self.layer2(xs).flatten(1)
        xs = xs.sigmoid()

        return xs
