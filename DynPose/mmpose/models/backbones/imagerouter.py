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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.pconv3 = Partial_conv3(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=1)
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.pconv5 = Partial_conv3(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=1)
        self.conv5_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.sigmoid = nn.Sigmoid()

        norm_cfg = dict(type='BN')
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, 32, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, 64, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, 64, postfix=3)
        self.norm4_name, norm4 = build_norm_layer(
            norm_cfg, 128, postfix=4)
        self.norm5_name, norm5 = build_norm_layer(
            norm_cfg, 128, postfix=5)
        self.norm6_name, norm6 = build_norm_layer(
            norm_cfg, 128, postfix=6)
        self.norm7_name, norm7 = build_norm_layer(
            norm_cfg, 64, postfix=7)
        self.norm8_name, norm8 = build_norm_layer(
            norm_cfg, 128, postfix=8)

        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)
        self.add_module(self.norm3_name, norm3)
        self.add_module(self.norm4_name, norm4)
        self.add_module(self.norm5_name, norm5)
        self.add_module(self.norm6_name, norm6)
        self.add_module(self.norm7_name, norm7)
        self.add_module(self.norm8_name, norm8)

        self.layer1 = nn.Conv2d(1536, 256, kernel_size=1, bias=True)
        self.layer2 = nn.Conv2d(256, 3, kernel_size=1, bias=True)

        self.sge1 = SpatialGroupEnhance(groups=8)
        self.sge2 = SpatialGroupEnhance(groups=8)

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

    @property
    def norm5(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm5_name)

    @property
    def norm6(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm6_name)

    @property
    def norm7(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm7_name)

    @property
    def norm8(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm8_name)

    def forward(self, x):
        """Forward function."""

        b = x.size(0)
        outs = []
        outs.append(x)
        # 转化为灰度图像
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.avg_pool(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.sge1(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3_1(x)
        x = self.norm7(x)
        x = self.relu(x)

        x = self.pconv3(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.sge2(x)
        x = self.norm4(x)
        x = self.relu(x)

        x = self.conv5_1(x)
        x = self.norm8(x)
        x = self.relu(x)

        x = self.pconv5(x)
        x = self.conv5(x)
        x = self.norm5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.norm6(x)
        x = self.relu(x)

        x = x.flatten()
        x = x.reshape(b, -1)

        x = self.layer1(x.unsqueeze(-1).unsqueeze(-1))
        x = F.relu(x, inplace=True)
        x = self.layer2(x).flatten(start_dim=1)
        score = x.sigmoid()
        outs.append(score)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)


import numpy as np
from torch.nn import init


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=8):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


class Partial_conv3(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim_conv3 = dim // 4
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x
