# Copyright (c) OpenMMLab. All rights reserved.
import copy
import types
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmdet.utils import ConfigType, reduce_mean
from mmengine.model import BaseModule, bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmpose.evaluation.functional import nms_torch
from mmpose.models.utils import (GAUEncoder, SinePositionalEncoding,
                                 filter_scores_and_topk)
from mmpose.registry import MODELS
from mmpose.structures.bbox import bbox_xyxy2cs
from mmpose.utils.typing import Features, OptSampleList, Predictions
from .yoloxpose_head_image import YOLOXPoseHeadImage

from .checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
                         find_latest_checkpoint, save_checkpoint,
                         weights_to_cpu)

import torch.nn.functional as F

EPS = 1e-8


class RTMOHeadModule(BaseModule):
    """RTMO head module for one-stage human pose estimation.

    This module predicts classification scores, bounding boxes, keypoint
    offsets and visibilities from multi-level feature maps.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        num_keypoints (int): Number of keypoints defined for one instance.
         in_channels (int): Number of channels in the input feature maps.
        cls_feat_channels (int): Number of channels in the classification score
            and objectness prediction branch. Defaults to 256.
         widen_factor (float): Width multiplier, multiply number of
             channels in each layer by this amount. Defaults to 1.0.
        num_groups (int): Group number of group convolution layers in keypoint
            regression branch. Defaults to 8.
        channels_per_group (int): Number of channels for each group of group
            convolution layers in keypoint regression branch. Defaults to 32.
        featmap_strides (Sequence[int]): Downsample factor of each feature
            map. Defaults to [8, 16, 32].
        conv_bias (bool or str): If specified as `auto`, it will be decided
            by the norm_cfg. Bias of conv will be set as True if `norm_cfg`
            is None, otherwise False. Defaults to "auto".
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        num_keypoints: int,
        in_channels: int,
        num_classes: int = 1,
        widen_factor: float = 1.0,
        cls_feat_channels: int = 256,
        stacked_convs: int = 2,
        num_groups=8,
        channels_per_group=36,
        pose_vec_channels=-1,
        featmap_strides: Sequence[int] = [8, 16, 32],
        conv_bias: Union[bool, str] = 'auto',
        conv_cfg: Optional[ConfigType] = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        init_cfg: Optional[ConfigType] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_feat_channels = int(cls_feat_channels * widen_factor)
        self.stacked_convs = stacked_convs
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.featmap_strides = featmap_strides

        self.in_channels = int(in_channels * widen_factor)
        self.num_keypoints = num_keypoints

        self.num_groups = num_groups
        self.channels_per_group = int(widen_factor * channels_per_group)
        self.pose_vec_channels = pose_vec_channels

        self.test_load = 0

        self._init_layers()

    def _init_layers(self):
        """Initialize heads for all level feature maps."""
        self._init_cls_branch()
        self._init_pose_branch()

    def _init_cls_branch(self):
        """Initialize classification branch for all level feature maps."""
        self.conv_cls = nn.ModuleList()
        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.cls_feat_channels
                stacked_convs.append(
                    ConvModule(
                        chn,
                        self.cls_feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias))
            self.conv_cls.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_cls = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_cls.append(
                nn.Conv2d(self.cls_feat_channels, self.num_classes, 1))

    def _init_pose_branch(self):
        """Initialize pose prediction branch for all level feature maps."""
        self.conv_pose = nn.ModuleList()
        out_chn = self.num_groups * self.channels_per_group
        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs * 2):
                chn = self.in_channels if i == 0 else out_chn
                groups = 1 if i == 0 else self.num_groups
                stacked_convs.append(
                    ConvModule(
                        chn,
                        out_chn,
                        3,
                        stride=1,
                        padding=1,
                        groups=groups,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias))
            self.conv_pose.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_bbox = nn.ModuleList()
        self.out_kpt_reg = nn.ModuleList()
        self.out_kpt_vis = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_bbox.append(nn.Conv2d(out_chn, 4, 1))
            self.out_kpt_reg.append(
                nn.Conv2d(out_chn, self.num_keypoints * 2, 1))
            self.out_kpt_vis.append(nn.Conv2d(out_chn, self.num_keypoints, 1))

        if self.pose_vec_channels > 0:
            self.out_pose = nn.ModuleList()
            for _ in self.featmap_strides:
                self.out_pose.append(
                    nn.Conv2d(out_chn, self.pose_vec_channels, 1))


    def init_weights(self):
        """Initialize weights of the head.

        Use prior in model initialization to improve stability.
        """

        super().init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv_cls in self.out_cls:
            conv_cls.bias.data.fill_(bias_init)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            cls_scores (List[Tensor]): Classification scores for each level.
            bbox_preds (List[Tensor]): Bounding box predictions for each level.
            kpt_offsets (List[Tensor]): Keypoint offsets for each level.
            kpt_vis (List[Tensor]): Keypoint visibilities for each level.
            pose_feats (List[Tensor]): Pose features for each level.
        """

        image = x[-1]
        x = x[0:-1]

        if self.is_load == 0:
            self.load_checkpoint("rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth", 'cpu', False)
            self.is_load = 1
            for k, v in self.conv_cls.named_parameters():
                if v.requires_grad == True:
                    v.requires_grad = False
            for k, v in self.conv_pose.named_parameters():
                if v.requires_grad == True:
                    v.requires_grad = False
            for k, v in self.out_cls.named_parameters():
                if v.requires_grad == True:
                    v.requires_grad = False
            for k, v in self.out_kpt_reg.named_parameters():
                if v.requires_grad == True:
                    v.requires_grad = False
            for k, v in self.out_kpt_vis.named_parameters():
                if v.requires_grad == True:
                    v.requires_grad = False
            for k, v in self.out_bbox.named_parameters():
                if v.requires_grad == True:
                    v.requires_grad = False
            for k, v in self.out_pose.named_parameters():
                if v.requires_grad == True:
                    v.requires_grad = False
            for k, v in self.net2.named_parameters():
                if v.requires_grad == True:
                    v.requires_grad = False

        if self.training:
            pass
        elif self.test_load == 0:
            self.load_checkpoint("rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth", 'cpu', False)
            self.test_load = 1
        else:
            pass


        router_result = self.router(x)

        if self.training:
            with torch.no_grad():
                cls_scores, bbox_preds = [], []
                kpt_offsets, kpt_vis = [], []
                pose_feats = []
                for i in range(len(x)):
                    cls_feat, reg_feat = x[i].split(x[i].size(1) // 2, 1)
                    cls_feat = self.conv_cls[i](cls_feat)
                    reg_feat = self.conv_pose[i](reg_feat)
                    cls_scores.append(self.out_cls[i](cls_feat))
                    bbox_preds.append(self.out_bbox[i](reg_feat))
                    if self.training:
                        # `kpt_offsets` generates the proxy poses for positive
                        # sample selection during training
                        kpt_offsets.append(self.out_kpt_reg[i](reg_feat))
                    kpt_vis.append(self.out_kpt_vis[i](reg_feat))

                    if self.pose_vec_channels > 0:
                        pose_feats.append(self.out_pose[i](reg_feat))
                    else:
                        pose_feats.append(reg_feat)

            with torch.no_grad():
                x_net2 = self.net2.backbone(image)
                x_net2 = self.net2.neck(x_net2)
                cls_scores_2, bbox_preds_2, kpt_offsets_2, kpt_vis_2, pose_feats_2 = self.net2.head.head_module(x_net2)
            # print('cls_scores: ', cls_scores[0].shape)
            # print('bbox_preds: ', bbox_preds[0].shape)
            # print('kpt_offsets: ', kpt_offsets[0].shape)
            # print('kpt_vis: ', kpt_vis[0].shape)
            # print('pose_feats: ', pose_feats[0].shape)

            # print('cls_scores_2: ', cls_scores_2[0].shape)
            # print('bbox_preds_2: ', bbox_preds_2[0].shape)
            # print('kpt_offsets_2: ', kpt_offsets_2[0].shape)
            # print('kpt_vis_2: ', kpt_vis_2[0].shape)
            # print('pose_feats_2: ', pose_feats_2[0].shape)
            #
            # print('router_result: ', router_result.shape)

            # return cls_scores, bbox_preds, kpt_offsets, kpt_vis, pose_feats, \
            #     cls_scores_2, bbox_preds_2, kpt_offsets_2, kpt_vis_2, pose_feats_2, (router_result.sum())/3.0
            # for k, v in self.conv_cls.named_parameters():
            #     print('self.conv_cls{}: {}'.format(k, v.requires_grad))
            # for k, v in self.conv_pose.named_parameters():
            #     print('self.conv_pose{}: {}'.format(k, v.requires_grad))
            # for k, v in self.out_kpt_reg.named_parameters():
            #     print('self.out_kpt_reg{}: {}'.format(k, v.requires_grad))
            # for k, v in self.out_kpt_vis.named_parameters():
            #     print('self.out_kpt_vis{}: {}'.format(k, v.requires_grad))
            # for k, v in self.out_kpt_vis.named_parameters():
            #     print('self.out_kpt_vis{}: {}'.format(k, v.requires_grad))
            # for k, v in self.net2.named_parameters():
            #     print('self.net2{}: {}'.format(k, v.requires_grad))
            return cls_scores, bbox_preds, kpt_offsets, kpt_vis, pose_feats, \
                cls_scores_2, bbox_preds_2, kpt_offsets_2, kpt_vis_2, pose_feats_2, router_result[0,0]
        else:
            # router_result[0, 0] = 0
            # print(router_result[0, 0])
            if router_result[0, 0] <= 0.50:
                # print("Net1")
                cls_scores, bbox_preds = [], []
                kpt_offsets, kpt_vis = [], []
                pose_feats = []
                # print(x[0].shape)
                # print(x[1].shape)
                for i in range(len(x)):

                    cls_feat, reg_feat = x[i].split(x[i].size(1) // 2, 1)

                    cls_feat = self.conv_cls[i](cls_feat)
                    reg_feat = self.conv_pose[i](reg_feat)

                    cls_scores.append(self.out_cls[i](cls_feat))
                    bbox_preds.append(self.out_bbox[i](reg_feat))
                    if self.training:
                        # `kpt_offsets` generates the proxy poses for positive
                        # sample selection during training
                        kpt_offsets.append(self.out_kpt_reg[i](reg_feat))
                    kpt_vis.append(self.out_kpt_vis[i](reg_feat))

                    if self.pose_vec_channels > 0:
                        pose_feats.append(self.out_pose[i](reg_feat))
                    else:
                        pose_feats.append(reg_feat)


                return cls_scores, bbox_preds, kpt_offsets, kpt_vis, pose_feats,1
                # return cls_scores_2, bbox_preds_2, kpt_offsets_2, kpt_vis_2, pose_feats_2

            else:
                # print("Net2")
                x_net2 = self.net2.backbone(image)
                x_net2 = self.net2.neck(x_net2)
                # print(x_net2[0].shape)
                # print(x_net2[1].shape)
                cls_scores_2, bbox_preds_2, kpt_offsets_2, kpt_vis_2, pose_feats_2 = self.net2.head.head_module(x_net2)
                return cls_scores_2, bbox_preds_2, kpt_offsets_2, kpt_vis_2, pose_feats_2, 2



    def load_checkpoint(self,
                        filename: str,
                        map_location,
                        strict,
                        revise_keys: list = [(r'^module.', '')]):
        """Load checkpoint from given ``filename``.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            map_location (str or callable): A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'cpu'.
            strict (bool): strict (bool): Whether to allow different params for
                the model and checkpoint.
            revise_keys (list): A list of customized keywords to modify the
                state_dict in checkpoint. Each item is a (pattern, replacement)
                pair of the regular expression operations. Defaults to strip
                the prefix 'module.' by [(r'^module\\.', '')].
        """
        checkpoint = _load_checkpoint(filename, map_location=map_location)
        model = self.net2
        checkpoint = _load_checkpoint_to_model(
            model, checkpoint, strict, revise_keys=revise_keys)
        print("rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth")

        return checkpoint


class DCC(BaseModule):
    """Dynamic Coordinate Classifier for One-stage Pose Estimation.

    Args:
        in_channels (int): Number of input feature map channels.
        num_keypoints (int): Number of keypoints for pose estimation.
        feat_channels (int): Number of feature channels.
        num_bins (Tuple[int, int]): Tuple representing the number of bins in
            x and y directions.
        spe_channels (int): Number of channels for Sine Positional Encoding.
            Defaults to 128.
        spe_temperature (float): Temperature for Sine Positional Encoding.
            Defaults to 300.0.
        gau_cfg (dict, optional): Configuration for Gated Attention Unit.
    """

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int,
        feat_channels: int,
        num_bins: Tuple[int, int],
        spe_channels: int = 128,
        spe_temperature: float = 300.0,
        gau_cfg: Optional[dict] = dict(
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc='add'),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_keypoints = num_keypoints

        self.feat_channels = feat_channels
        self.num_bins = num_bins
        self.gau_cfg = gau_cfg

        self.spe = SinePositionalEncoding(
            out_channels=spe_channels,
            temperature=spe_temperature,
        )
        self.spe_feat_channels = spe_channels

        self._build_layers()
        self._build_basic_bins()

    def _build_layers(self):
        """Builds layers for the model."""

        # GAU encoder
        if self.gau_cfg is not None:
            gau_cfg = self.gau_cfg.copy()
            gau_cfg['in_token_dims'] = self.feat_channels
            gau_cfg['out_token_dims'] = self.feat_channels
            self.gau = GAUEncoder(**gau_cfg)
            if gau_cfg.get('pos_enc', 'none') in ('add', 'rope'):
                self.pos_enc = nn.Parameter(
                    torch.randn(self.num_keypoints, gau_cfg['s']))

        # fully-connected layers to convert pose feats to keypoint feats
        pose_to_kpts = [
            nn.Linear(self.in_channels,self.feat_channels * self.num_keypoints),
            nn.BatchNorm1d(self.feat_channels * self.num_keypoints)
        ]
        self.pose_to_kpts = nn.Sequential(*pose_to_kpts)

        # adapter layers for dynamic encodings
        self.x_fc = nn.Linear(self.spe_feat_channels, self.feat_channels)
        self.y_fc = nn.Linear(self.spe_feat_channels, self.feat_channels)

        # fully-connected layers to predict sigma
        self.sigma_fc = nn.Sequential(
            nn.Linear(self.in_channels, self.num_keypoints), nn.Sigmoid(),
            Scale(0.1))

    def _build_basic_bins(self):
        """Builds basic bin coordinates for x and y."""
        self.register_buffer('y_bins',
                             torch.linspace(-0.5, 0.5, self.num_bins[1]))
        self.register_buffer('x_bins',
                             torch.linspace(-0.5, 0.5, self.num_bins[0]))

    def _apply_softmax(self, x_hms, y_hms):
        """Apply softmax on 1-D heatmaps.

        Args:
            x_hms (Tensor): 1-D heatmap in x direction.
            y_hms (Tensor): 1-D heatmap in y direction.

        Returns:
            tuple: A tuple containing the normalized x and y heatmaps.
        """

        x_hms = x_hms.clamp(min=-5e4, max=5e4)
        y_hms = y_hms.clamp(min=-5e4, max=5e4)
        pred_x = x_hms - x_hms.max(dim=-1, keepdims=True).values.detach()
        pred_y = y_hms - y_hms.max(dim=-1, keepdims=True).values.detach()

        exp_x, exp_y = pred_x.exp(), pred_y.exp()
        prob_x = exp_x / (exp_x.sum(dim=-1, keepdims=True) + EPS)
        prob_y = exp_y / (exp_y.sum(dim=-1, keepdims=True) + EPS)

        return prob_x, prob_y

    def _get_bin_enc(self, bbox_cs, grids):
        """Calculate dynamic bin encodings for expanded bounding box.

        This function computes dynamic bin allocations and encodings based
        on the expanded bounding box center-scale (bbox_cs) and grid values.
        The process involves adjusting the bins according to the scale and
        center of the bounding box and then applying a sinusoidal positional
        encoding (spe) followed by a fully connected layer (fc) to obtain the
        final x and y bin encodings.

        Args:
            bbox_cs (Tensor): A tensor representing the center and scale of
                bounding boxes.
            grids (Tensor): A tensor representing the grid coordinates.

        Returns:
            tuple: A tuple containing the encoded x and y bins.
        """
        center, scale = bbox_cs.split(2, dim=-1)
        center = center - grids

        x_bins, y_bins = self.x_bins, self.y_bins

        # dynamic bin allocation
        x_bins = x_bins.view(*((1,) * (scale.ndim-1)), -1) \
            * scale[..., 0:1] + center[..., 0:1]
        y_bins = y_bins.view(*((1,) * (scale.ndim-1)), -1) \
            * scale[..., 1:2] + center[..., 1:2]

        # dynamic bin encoding
        x_bins_enc = self.x_fc(self.spe(position=x_bins))
        y_bins_enc = self.y_fc(self.spe(position=y_bins))

        return x_bins_enc, y_bins_enc

    def _pose_feats_to_heatmaps(self, pose_feats, x_bins_enc, y_bins_enc):
        """Convert pose features to heatmaps using x and y bin encodings.

        This function transforms the given pose features into keypoint
        features and then generates x and y heatmaps based on the x and y
        bin encodings. If Gated attention unit (gau) is used, it applies it
        to the keypoint features. The heatmaps are generated using matrix
        multiplication of pose features and bin encodings.

        Args:
            pose_feats (Tensor): The pose features tensor.
            x_bins_enc (Tensor): The encoded x bins tensor.
            y_bins_enc (Tensor): The encoded y bins tensor.

        Returns:
            tuple: A tuple containing the x and y heatmaps.
        """
        # print(pose_feats.shape)
        if len(pose_feats) == 1:
            pose_feats_new = torch.cat([pose_feats, pose_feats])
            kpt_feats_double = self.pose_to_kpts(pose_feats_new)
            kpt_feats = kpt_feats_double[0].unsqueeze(0)
        else:
            kpt_feats = self.pose_to_kpts(pose_feats)

        kpt_feats = kpt_feats.reshape(*kpt_feats.shape[:-1],self.num_keypoints, self.feat_channels)

        if hasattr(self, 'gau'):
            kpt_feats = self.gau(
                kpt_feats, pos_enc=getattr(self, 'pos_enc', None))

        x_hms = torch.matmul(kpt_feats,x_bins_enc.transpose(-1, -2).contiguous())
        y_hms = torch.matmul(kpt_feats,y_bins_enc.transpose(-1, -2).contiguous())

        return x_hms, y_hms

    def _decode_xy_heatmaps(self, x_hms, y_hms, bbox_cs):
        """Decode x and y heatmaps to obtain coordinates.

        This function  decodes x and y heatmaps to obtain the corresponding
        coordinates. It adjusts the x and y bins based on the bounding box
        center and scale, and then computes the weighted sum of these bins
        with the heatmaps to derive the x and y coordinates.

        Args:
            x_hms (Tensor): The normalized x heatmaps tensor.
            y_hms (Tensor): The normalized y heatmaps tensor.
            bbox_cs (Tensor): The bounding box center-scale tensor.

        Returns:
            Tensor: A tensor of decoded x and y coordinates.
        """
        center, scale = bbox_cs.split(2, dim=-1)

        x_bins, y_bins = self.x_bins, self.y_bins

        x_bins = x_bins.view(*((1,) * (scale.ndim-1)), -1) \
            * scale[..., 0:1] + center[..., 0:1]
        y_bins = y_bins.view(*((1,) * (scale.ndim-1)), -1) \
            * scale[..., 1:2] + center[..., 1:2]

        x = (x_hms * x_bins.unsqueeze(1)).sum(dim=-1)
        y = (y_hms * y_bins.unsqueeze(1)).sum(dim=-1)

        return torch.stack((x, y), dim=-1)

    def generate_target_heatmap(self, kpt_targets, bbox_cs, sigmas, areas):
        """Generate target heatmaps for keypoints based on bounding box.

        This function calculates x and y bins adjusted by bounding box center
        and scale. It then computes distances from keypoint targets to these
        bins and normalizes these distances based on the areas and sigmas.
        Finally, it uses these distances to generate heatmaps for x and y
        coordinates under assumption of laplacian error.

        Args:
            kpt_targets (Tensor): Keypoint targets tensor.
            bbox_cs (Tensor): Bounding box center-scale tensor.
            sigmas (Tensor): Learned deviation of grids.
            areas (Tensor): Areas of GT instance assigned to grids.

        Returns:
            tuple: A tuple containing the x and y heatmaps.
        """

        # calculate the error of each bin from the GT keypoint coordinates
        center, scale = bbox_cs.split(2, dim=-1)
        x_bins = self.x_bins.view(*((1,) * (scale.ndim-1)), -1) \
            * scale[..., 0:1] + center[..., 0:1]
        y_bins = self.y_bins.view(*((1,) * (scale.ndim-1)), -1) \
            * scale[..., 1:2] + center[..., 1:2]

        dist_x = torch.abs(kpt_targets.narrow(2, 0, 1) - x_bins.unsqueeze(1))
        dist_y = torch.abs(kpt_targets.narrow(2, 1, 1) - y_bins.unsqueeze(1))

        # normalize
        areas = areas.pow(0.5).clip(min=1).reshape(-1, 1, 1)
        sigmas = sigmas.clip(min=1e-3).unsqueeze(2)
        dist_x = dist_x / areas / sigmas
        dist_y = dist_y / areas / sigmas

        hm_x = torch.exp(-dist_x / 2) / sigmas
        hm_y = torch.exp(-dist_y / 2) / sigmas

        return hm_x, hm_y

    def forward_train(self, pose_feats, bbox_cs, grids):
        """Forward pass for training.

        This function processes pose features during training. It computes
        sigmas using a fully connected layer, generates bin encodings,
        creates heatmaps from pose features, applies softmax to the heatmaps,
        and then decodes the heatmaps to get pose predictions.

        Args:
            pose_feats (Tensor): The pose features tensor.
            bbox_cs (Tensor): The bounding box in the format of center & scale.
            grids (Tensor): The grid coordinates.

        Returns:
            tuple: A tuple containing pose predictions, heatmaps, and sigmas.
        """
        with torch.no_grad():
            sigmas = self.sigma_fc(pose_feats)
            x_bins_enc, y_bins_enc = self._get_bin_enc(bbox_cs, grids)
            x_hms, y_hms = self._pose_feats_to_heatmaps(pose_feats, x_bins_enc,y_bins_enc)
            x_hms, y_hms = self._apply_softmax(x_hms, y_hms)
            pose_preds = self._decode_xy_heatmaps(x_hms, y_hms, bbox_cs)
        return pose_preds, (x_hms, y_hms), sigmas

    @torch.no_grad()
    def forward_test(self, pose_feats, bbox_cs, grids):
        """Forward pass for testing.

        This function processes pose features during testing. It generates
        bin encodings, creates heatmaps from pose features, and then decodes
        the heatmaps to get pose predictions.

        Args:
            pose_feats (Tensor): The pose features tensor.
            bbox_cs (Tensor): The bounding box in the format of center & scale.
            grids (Tensor): The grid coordinates.

        Returns:
            Tensor: Pose predictions tensor.
        """
        x_bins_enc, y_bins_enc = self._get_bin_enc(bbox_cs, grids)
        x_hms, y_hms = self._pose_feats_to_heatmaps(pose_feats, x_bins_enc,
                                                    y_bins_enc)
        x_hms, y_hms = self._apply_softmax(x_hms, y_hms)
        pose_preds = self._decode_xy_heatmaps(x_hms, y_hms, bbox_cs)
        return pose_preds

    def switch_to_deploy(self, test_cfg: Optional[Dict] = None):
        if getattr(self, 'deploy', False):
            return

        self._convert_pose_to_kpts()
        if hasattr(self, 'gau'):
            self._convert_gau()
        self._convert_forward_test()

        self.deploy = True

    def _convert_pose_to_kpts(self):
        """Merge BatchNorm layer into Fully Connected layer.

        This function merges a BatchNorm layer into the associated Fully
        Connected layer to avoid dimension mismatch during ONNX exportation. It
        adjusts the weights and biases of the FC layer to incorporate the BN
        layer's parameters, and then replaces the original FC layer with the
        updated one.
        """
        fc, bn = self.pose_to_kpts

        # Calculate adjusted weights and biases
        std = (bn.running_var + bn.eps).sqrt()
        weight = fc.weight * (bn.weight / std).unsqueeze(1)
        bias = bn.bias + (fc.bias - bn.running_mean) * bn.weight / std

        # Update FC layer with adjusted parameters
        fc.weight.data = weight.detach()
        fc.bias.data = bias.detach()
        self.pose_to_kpts = fc

    def _convert_gau(self):
        """Reshape and merge tensors for Gated Attention Unit (GAU).

        This function pre-processes the gamma and beta tensors of the GAU and
        handles the position encoding if available. It also redefines the GAU's
        forward method to incorporate these pre-processed tensors, optimizing
        the computation process.
        """
        # Reshape gamma and beta tensors in advance
        gamma_q = self.gau.gamma[0].view(1, 1, 1, self.gau.gamma.size(-1))
        gamma_k = self.gau.gamma[1].view(1, 1, 1, self.gau.gamma.size(-1))
        beta_q = self.gau.beta[0].view(1, 1, 1, self.gau.beta.size(-1))
        beta_k = self.gau.beta[1].view(1, 1, 1, self.gau.beta.size(-1))

        # Adjust beta tensors with position encoding if available
        if hasattr(self, 'pos_enc'):
            pos_enc = self.pos_enc.reshape(1, 1, *self.pos_enc.shape)
            beta_q = beta_q + pos_enc
            beta_k = beta_k + pos_enc

        gamma_q = gamma_q.detach().cpu()
        gamma_k = gamma_k.detach().cpu()
        beta_q = beta_q.detach().cpu()
        beta_k = beta_k.detach().cpu()

        @torch.no_grad()
        def _forward(self, x, *args, **kwargs):
            norm = torch.linalg.norm(x, dim=-1, keepdim=True) * self.ln.scale
            x = x / norm.clamp(min=self.ln.eps) * self.ln.g

            uv = self.uv(x)
            uv = self.act_fn(uv)

            u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=-1)
            if not torch.onnx.is_in_onnx_export():
                q = base * gamma_q.to(base) + beta_q.to(base)
                k = base * gamma_k.to(base) + beta_k.to(base)
            else:
                q = base * gamma_q + beta_q
                k = base * gamma_k + beta_k
            qk = torch.matmul(q, k.transpose(-1, -2))

            kernel = torch.square(torch.nn.functional.relu(qk / self.sqrt_s))
            x = u * torch.matmul(kernel, v)
            x = self.o(x)
            return x

        self.gau._forward = types.MethodType(_forward, self.gau)

    def _convert_forward_test(self):
        """Simplify the forward test process.

        This function precomputes certain tensors and redefines the
        forward_test method for the model. It includes steps for converting
        pose features to keypoint features, performing dynamic bin encoding,
        calculating 1-D heatmaps, and decoding these heatmaps to produce final
        pose predictions.
        """
        x_bins_ = self.x_bins.view(1, 1, -1).detach().cpu()
        y_bins_ = self.y_bins.view(1, 1, -1).detach().cpu()
        dim_t = self.spe.dim_t.view(1, 1, 1, -1).detach().cpu()

        @torch.no_grad()
        def _forward_test(self, pose_feats, bbox_cs, grids):

            # step 1: pose features -> keypoint features

            kpt_feats = self.pose_to_kpts(pose_feats)
            kpt_feats = kpt_feats.reshape(*kpt_feats.shape[:-1],
                                          self.num_keypoints,
                                          self.feat_channels)
            kpt_feats = self.gau(kpt_feats)

            # step 2: dynamic bin encoding
            center, scale = bbox_cs.split(2, dim=-1)
            center = center - grids

            if not torch.onnx.is_in_onnx_export():
                x_bins = x_bins_.to(scale) * scale[..., 0:1] + center[..., 0:1]
                y_bins = y_bins_.to(scale) * scale[..., 1:2] + center[..., 1:2]
                freq_x = x_bins.unsqueeze(-1) / dim_t.to(scale)
                freq_y = y_bins.unsqueeze(-1) / dim_t.to(scale)
            else:
                x_bins = x_bins_ * scale[..., 0:1] + center[..., 0:1]
                y_bins = y_bins_ * scale[..., 1:2] + center[..., 1:2]
                freq_x = x_bins.unsqueeze(-1) / dim_t
                freq_y = y_bins.unsqueeze(-1) / dim_t

            spe_x = torch.cat((freq_x.cos(), freq_x.sin()), dim=-1)
            spe_y = torch.cat((freq_y.cos(), freq_y.sin()), dim=-1)

            x_bins_enc = self.x_fc(spe_x).transpose(-1, -2).contiguous()
            y_bins_enc = self.y_fc(spe_y).transpose(-1, -2).contiguous()

            # step 3: calculate 1-D heatmaps
            x_hms = torch.matmul(kpt_feats, x_bins_enc)
            y_hms = torch.matmul(kpt_feats, y_bins_enc)
            x_hms, y_hms = self._apply_softmax(x_hms, y_hms)

            # step 4: decode 1-D heatmaps through integral
            x = (x_hms * x_bins.unsqueeze(-2)).sum(dim=-1) + grids[..., 0:1]
            y = (y_hms * y_bins.unsqueeze(-2)).sum(dim=-1) + grids[..., 1:2]

            keypoints = torch.stack((x, y), dim=-1)

            if not torch.onnx.is_in_onnx_export():
                keypoints = keypoints.squeeze(0)
            return keypoints

        self.forward_test = types.MethodType(_forward_test, self)


@MODELS.register_module()
class RTMOHeadImage(YOLOXPoseHeadImage):
    """One-stage coordinate classification head introduced in RTMO (2023). This
    head incorporates dynamic coordinate classification and YOLO structure for
    precise keypoint localization.

    Args:
        num_keypoints (int): Number of keypoints to detect.
        head_module_cfg (ConfigType): Configuration for the head module.
        featmap_strides (Sequence[int]): Strides of feature maps.
            Defaults to [16, 32].
        num_classes (int): Number of object classes, defaults to 1.
        use_aux_loss (bool): Indicates whether to use auxiliary loss,
            defaults to False.
        proxy_target_cc (bool): Indicates whether to use keypoints predicted
            by coordinate classification as the targets for proxy regression
            branch. Defaults to False.
        assigner (ConfigType): Configuration for positive sample assigning
            module.
        prior_generator (ConfigType): Configuration for prior generation.
        bbox_padding (float): Padding for bounding boxes, defaults to 1.25.
        overlaps_power (float): Power factor adopted by overlaps before they
            are assigned as targets in classification loss. Defaults to 1.0.
        dcc_cfg (Optional[ConfigType]): Configuration for dynamic coordinate
            classification module.
        loss_cls (Optional[ConfigType]): Configuration for classification loss.
        loss_bbox (Optional[ConfigType]): Configuration for bounding box loss.
        loss_oks (Optional[ConfigType]): Configuration for OKS loss.
        loss_vis (Optional[ConfigType]): Configuration for visibility loss.
        loss_mle (Optional[ConfigType]): Configuration for MLE loss.
        loss_bbox_aux (Optional[ConfigType]): Configuration for auxiliary
            bounding box loss.
    """

    def __init__(
        self,
        num_keypoints: int,
        head_module_cfg: ConfigType,
        featmap_strides: Sequence[int] = [16, 32],
        num_classes: int = 1,
        use_aux_loss: bool = False,
        proxy_target_cc: bool = False,
        assigner: ConfigType = None,
        prior_generator: ConfigType = None,
        bbox_padding: float = 1.25,
        overlaps_power: float = 1.0,
        dcc_cfg: Optional[ConfigType] = None,
        loss_cls: Optional[ConfigType] = None,
        loss_bbox: Optional[ConfigType] = None,
        loss_oks: Optional[ConfigType] = None,
        loss_vis: Optional[ConfigType] = None,
        loss_mle: Optional[ConfigType] = None,
        loss_bbox_aux: Optional[ConfigType] = None,
    ):
        super().__init__(
            num_keypoints=num_keypoints,
            head_module_cfg=None,
            featmap_strides=featmap_strides,
            num_classes=num_classes,
            use_aux_loss=use_aux_loss,
            assigner=assigner,
            prior_generator=prior_generator,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_oks=loss_oks,
            loss_vis=loss_vis,
            loss_bbox_aux=loss_bbox_aux,
            overlaps_power=overlaps_power)

        self.bbox_padding = bbox_padding

        # override to ensure consistency
        head_module_cfg['featmap_strides'] = featmap_strides
        head_module_cfg['num_keypoints'] = num_keypoints

        # build modules
        self.head_module = RTMOHeadModule(**head_module_cfg)

        self.proxy_target_cc = proxy_target_cc
        if dcc_cfg is not None:
            dcc_cfg['num_keypoints'] = num_keypoints
            self.dcc = DCC(**dcc_cfg)

        for k, v in self.dcc.named_parameters():
            if v.requires_grad == True:
                v.requires_grad = False

        # build losses
        if loss_mle is not None:
            self.loss_mle = MODELS.build(loss_mle)

        metafile = 'configs/_base_/datasets/coco.py'
        widen_factor = 1.0
        deepen_factor = 1.0
        network2 = dict(
            type='BottomupPoseEstimator',
            backbone=dict(
                type='CSPDarknet',
                frozen_stages=4,
                deepen_factor=deepen_factor,
                widen_factor=widen_factor,
                out_indices=(2, 3, 4),
                spp_kernal_sizes=(5, 9, 13),
                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg=dict(type='Swish'),
                ),
            neck=dict(
                type='HybridEncoder',
                in_channels=[256, 512, 1024],
                deepen_factor=deepen_factor,
                widen_factor=widen_factor,
                hidden_dim=256,
                output_indices=[1, 2],
                encoder_cfg=dict(
                    self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        ffn_drop=0.0,
                        act_cfg=dict(type='GELU'))),
                projector=dict(
                    type='ChannelMapper',
                    in_channels=[256, 256],
                    kernel_size=1,
                    out_channels=512,
                    act_cfg=None,
                    norm_cfg=dict(type='BN'),
                    num_outs=2)),
            head=dict(
                type='RTMOHead',
                num_keypoints=17,
                featmap_strides=(16, 32),
                head_module_cfg=dict(
                    num_classes=1,
                    in_channels=256,
                    cls_feat_channels=256,
                    channels_per_group=36,
                    pose_vec_channels=512,
                    widen_factor=widen_factor,
                    stacked_convs=2,
                    norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                    act_cfg=dict(type='Swish')),
                assigner=dict(
                    type='SimOTAAssigner',
                    dynamic_k_indicator='oks',
                    oks_calculator=dict(type='PoseOKS', metainfo=metafile)),
                prior_generator=dict(
                    type='MlvlPointGenerator',
                    centralize_points=True,
                    strides=[16, 32]),
                dcc_cfg=dict(
                    in_channels=512,
                    feat_channels=128,
                    num_bins=(192, 256),
                    spe_channels=128,
                    gau_cfg=dict(
                        s=128,
                        expansion_factor=2,
                        dropout_rate=0.0,
                        drop_path=0.0,
                        act_fn='SiLU',
                        pos_enc='add')),
                overlaps_power=0.5,
                loss_cls=dict(
                    type='VariFocalLoss',
                    reduction='sum',
                    use_target_weight=True,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='IoULoss',
                    mode='square',
                    eps=1e-16,
                    reduction='sum',
                    loss_weight=5.0),
                loss_oks=dict(
                    type='OKSLoss',
                    reduction='none',
                    metainfo=metafile,
                    loss_weight=30.0),
                loss_vis=dict(
                    type='BCELoss',
                    use_target_weight=True,
                    reduction='mean',
                    loss_weight=1.0),
                loss_mle=dict(
                    type='MLECCLoss',
                    use_target_weight=True,
                    loss_weight=1e-2,
                ),
                loss_bbox_aux=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
            ))

        self.head_module.net2 = MODELS.build(network2)
        self.head_module.is_load = 0
        self.head_module.router = AdaptiveRouter([256, 256], 1, reduction=4)


    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """

        # 1. collect & reform predictions
        cls_scores, bbox_preds, kpt_offsets, kpt_vis, pose_vecs, cls_scores_2, bbox_preds_2, kpt_offsets_2, kpt_vis_2, pose_vecs_2, score = self.forward(feats)
        # for net1
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
        flatten_priors = torch.cat(mlvl_priors)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = self._flatten_predictions(cls_scores)
        flatten_bbox_preds = self._flatten_predictions(bbox_preds)
        flatten_objectness = torch.ones_like(
            flatten_cls_scores).detach().narrow(-1, 0, 1) * 1e4
        flatten_kpt_offsets = self._flatten_predictions(kpt_offsets)
        flatten_kpt_vis = self._flatten_predictions(kpt_vis)
        flatten_pose_vecs = self._flatten_predictions(pose_vecs)
        flatten_bbox_decoded = self.decode_bbox(flatten_bbox_preds,
                                                flatten_priors[..., :2],
                                                flatten_priors[..., -1])
        flatten_kpt_decoded = self.decode_kpt_reg(flatten_kpt_offsets,
                                                  flatten_priors[..., :2],
                                                  flatten_priors[..., -1])

        # for net2
        featmap_sizes_2 = [cls_score.shape[2:] for cls_score in cls_scores_2]
        mlvl_priors_2 = self.prior_generator.grid_priors(
            featmap_sizes_2,
            dtype=cls_scores_2[0].dtype,
            device=cls_scores_2[0].device,
            with_stride=True)
        flatten_priors_2 = torch.cat(mlvl_priors_2)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores_2 = self.head_module.net2.head._flatten_predictions(cls_scores_2)
        flatten_bbox_preds_2 = self.head_module.net2.head._flatten_predictions(bbox_preds_2)
        flatten_objectness_2 = torch.ones_like(
            flatten_cls_scores_2).detach().narrow(-1, 0, 1) * 1e4
        flatten_kpt_offsets_2 = self.head_module.net2.head._flatten_predictions(kpt_offsets_2)
        flatten_kpt_vis_2 = self.head_module.net2.head._flatten_predictions(kpt_vis_2)
        flatten_pose_vecs_2 = self.head_module.net2.head._flatten_predictions(pose_vecs_2)
        flatten_bbox_decoded_2 = self.head_module.net2.head.decode_bbox(flatten_bbox_preds_2,
                                                flatten_priors_2[..., :2],
                                                flatten_priors_2[..., -1])
        flatten_kpt_decoded_2 = self.head_module.net2.head.decode_kpt_reg(flatten_kpt_offsets_2,
                                                  flatten_priors_2[..., :2],
                                                  flatten_priors_2[..., -1])

        # 2. generate targets for net 1
        targets = self._get_targets(flatten_priors,
                                    flatten_cls_scores.detach(),
                                    flatten_objectness.detach(),
                                    flatten_bbox_decoded.detach(),
                                    flatten_kpt_decoded.detach(),
                                    flatten_kpt_vis.detach(),
                                    batch_data_samples)
        pos_masks, cls_targets, obj_targets, obj_weights, \
            bbox_targets, bbox_aux_targets, kpt_targets, kpt_aux_targets, \
            vis_targets, vis_weights, pos_areas, pos_priors, group_indices, \
            num_fg_imgs = targets

        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_scores.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        # for net2
        targets_2 = self.head_module.net2.head._get_targets(flatten_priors_2,
                                    flatten_cls_scores_2.detach(),
                                    flatten_objectness_2.detach(),
                                    flatten_bbox_decoded_2.detach(),
                                    flatten_kpt_decoded_2.detach(),
                                    flatten_kpt_vis_2.detach(),
                                    batch_data_samples)
        pos_masks_2, cls_targets_2, obj_targets_2, obj_weights_2, \
            bbox_targets_2, bbox_aux_targets_2, kpt_targets_2, kpt_aux_targets_2, \
            vis_targets_2, vis_weights_2, pos_areas_2, pos_priors_2, group_indices_2, \
            num_fg_imgs_2 = targets_2

        num_pos_2 = torch.tensor(
            sum(num_fg_imgs_2),
            dtype=torch.float,
            device=flatten_cls_scores_2.device)
        num_total_samples_2 = max(reduce_mean(num_pos_2), 1.0)

        # 3. calculate loss for net1
        # extra_info = dict(num_samples=num_total_samples)
        losses = dict()
        cls_preds_all = flatten_cls_scores.view(-1, self.num_classes)
        cls_preds_all_2 = flatten_cls_scores_2.view(-1, self.num_classes)

        if num_pos > 0:

            # 3.1 bbox loss
            bbox_preds = flatten_bbox_decoded.view(-1, 4)[pos_masks]
            bbox_preds_2 = flatten_bbox_decoded_2.view(-1, 4)[pos_masks_2]
            loss_box1 = self.loss_bbox(bbox_preds, bbox_targets)/ num_total_samples
            loss_box2 = self.loss_bbox(bbox_preds_2, bbox_targets_2)/ num_total_samples_2
            delt_box = loss_box1 - loss_box2
            losses['loss_bbox'] = (1-score)*(loss_box1 - delt_box*0.5)+score*(loss_box2 + delt_box*0.5)
            # print(self.loss_bbox(bbox_preds, bbox_targets) / num_total_samples)

            if self.use_aux_loss:
                if hasattr(self, 'loss_bbox_aux'):
                    bbox_preds_raw = flatten_bbox_preds.view(-1, 4)[pos_masks]
                    bbox_preds_raw_2 = flatten_bbox_preds_2.view(-1, 4)[pos_masks_2]
                    loss_bbox_aux_1 = self.loss_bbox_aux(bbox_preds_raw, bbox_aux_targets) / num_total_samples
                    loss_bbox_aux_2 = self.loss_bbox_aux(bbox_preds_raw_2, bbox_aux_targets_2) / num_total_samples_2
                    delt_box_aux_1 = loss_bbox_aux_1 - loss_bbox_aux_2
                    losses['loss_bbox_aux'] = (1-score)*(loss_bbox_aux_1 - delt_box_aux_1*0.5)+score*(loss_bbox_aux_2 + delt_box_aux_1*0.5)

            # 3.2 keypoint visibility loss
            kpt_vis_preds = flatten_kpt_vis.view(-1,self.num_keypoints)[pos_masks]
            kpt_vis_preds_2 = flatten_kpt_vis_2.view(-1, self.num_keypoints)[pos_masks_2]
            loss_vis_1 = self.loss_vis(kpt_vis_preds, vis_targets,vis_weights)
            loss_vis_2 = self.loss_vis(kpt_vis_preds_2, vis_targets_2,vis_weights_2)
            delt_vis = loss_vis_1 - loss_vis_2
            losses['loss_vis'] = (1-score)*(loss_vis_1 - delt_vis*0.5)+score*(loss_vis_2 + delt_vis*0.5)

            # 3.3 keypoint loss
            kpt_reg_preds = flatten_kpt_decoded.view(-1, self.num_keypoints,2)[pos_masks]
            kpt_reg_preds_2 = flatten_kpt_decoded_2.view(-1, self.num_keypoints, 2)[pos_masks_2]

            if hasattr(self, 'loss_mle') and self.loss_mle.loss_weight > 0:
                pose_vecs = flatten_pose_vecs.view(
                    -1, flatten_pose_vecs.size(-1))[pos_masks]
                bbox_cs = torch.cat(bbox_xyxy2cs(bbox_preds, self.bbox_padding), dim=1)
                # 'cc' refers to 'cordinate classification'
                kpt_cc_preds, pred_hms, sigmas = self.dcc.forward_train(pose_vecs,bbox_cs,pos_priors[..., :2])
                target_hms = self.dcc.generate_target_heatmap(kpt_targets, bbox_cs, sigmas, pos_areas)
                loss_mle_1 = self.loss_mle(pred_hms, target_hms,vis_targets)
                losses['loss_mle'] = loss_mle_1

            if self.proxy_target_cc:
                # form the regression target using the coordinate
                # classification predictions

                diff_cc = torch.norm(kpt_cc_preds - kpt_targets, dim=-1)
                diff_reg = torch.norm(kpt_reg_preds - kpt_targets, dim=-1)
                mask = (diff_reg > diff_cc).float()
                kpt_weights_reg = vis_targets * mask
                oks = self.assigner.oks_calculator(kpt_cc_preds,
                                                   kpt_targets,
                                                   vis_targets, pos_areas)
                cls_targets = oks.unsqueeze(1)

                loss_oks_1 = self.loss_oks(kpt_reg_preds,
                                                   kpt_cc_preds.detach(),
                                                   kpt_weights_reg, pos_areas)
                losses['loss_oks'] = loss_oks_1

            else:
                loss_oks_1 = self.loss_oks(kpt_reg_preds, kpt_targets,vis_targets, pos_areas)
                losses['loss_oks'] = loss_oks_1

            cls_targets = cls_targets.pow(self.overlaps_power).detach()
            obj_targets[pos_masks] = cls_targets.to(obj_targets)

            cls_targets_2 = cls_targets_2.pow(self.head_module.net2.head.overlaps_power).detach()
            obj_targets_2[pos_masks_2] = cls_targets_2.to(obj_targets_2)

        # 3.4 classification loss
        loss_cls_1 = self.loss_cls(cls_preds_all, obj_targets,obj_weights) / num_total_samples
        loss_cls_2 = self.loss_cls(cls_preds_all_2, obj_targets_2, obj_weights_2) / num_total_samples_2
        delt_cls = loss_cls_1 - loss_cls_2
        losses['loss_cls'] = (1-score)*(loss_cls_1 - delt_cls*0.5)+score*(loss_cls_2 + delt_cls*0.5)

        return losses

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-scale features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (1, h, w)
                    or (K+1, h, w) if keypoint heatmaps are predicted
                - displacements (Tensor): The predicted displacement fields
                    in shape (K*2, h, w)
        """

        cls_scores, bbox_preds, _, kpt_vis, pose_vecs,flag = self.forward(feats)

        cfg = copy.deepcopy(test_cfg)

        batch_img_metas = [d.metainfo for d in batch_data_samples]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full((featmap_size.numel(), ),
                                    stride) for featmap_size, stride in zip(
                                        featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten predictions
        flatten_cls_scores = self._flatten_predictions(cls_scores).sigmoid()
        flatten_bbox_preds = self._flatten_predictions(bbox_preds)
        flatten_kpt_vis = self._flatten_predictions(kpt_vis).sigmoid()
        flatten_pose_vecs = self._flatten_predictions(pose_vecs)
        if flatten_pose_vecs is None:
            flatten_pose_vecs = [None] * len(batch_img_metas)
        flatten_bbox_preds = self.decode_bbox(flatten_bbox_preds,
                                              flatten_priors, flatten_stride)

        results_list = []
        for (bboxes, scores, kpt_vis, pose_vecs,
             img_meta) in zip(flatten_bbox_preds, flatten_cls_scores,
                              flatten_kpt_vis, flatten_pose_vecs,
                              batch_img_metas):

            score_thr = cfg.get('score_thr', 0.01)

            nms_pre = cfg.get('nms_pre', 100000)
            scores, labels = scores.max(1, keepdim=True)
            scores, _, keep_idxs_score, results = filter_scores_and_topk(
                scores, score_thr, nms_pre, results=dict(labels=labels[:, 0]))
            labels = results['labels']

            bboxes = bboxes[keep_idxs_score]
            kpt_vis = kpt_vis[keep_idxs_score]
            grids = flatten_priors[keep_idxs_score]
            stride = flatten_stride[keep_idxs_score]

            if bboxes.numel() > 0:
                nms_thr = cfg.get('nms_thr', 1.0)
                if nms_thr < 1.0:

                    keep_idxs_nms = nms_torch(bboxes, scores, nms_thr)
                    bboxes = bboxes[keep_idxs_nms]
                    stride = stride[keep_idxs_nms]
                    labels = labels[keep_idxs_nms]
                    kpt_vis = kpt_vis[keep_idxs_nms]
                    scores = scores[keep_idxs_nms]

                pose_vecs = pose_vecs[keep_idxs_score][keep_idxs_nms]
                bbox_cs = torch.cat(
                    bbox_xyxy2cs(bboxes, self.bbox_padding), dim=1)
                grids = grids[keep_idxs_nms]
                if flag == 1:
                    keypoints = self.dcc.forward_test(pose_vecs, bbox_cs, grids)
                elif flag == 2:
                    keypoints = self.head_module.net2.head.dcc.forward_test(pose_vecs, bbox_cs, grids)

            else:
                # empty prediction
                keypoints = bboxes.new_zeros((0, self.num_keypoints, 2))

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes,
                bbox_scores=scores,
                keypoints=keypoints,
                keypoint_scores=kpt_vis,
                keypoints_visible=kpt_vis)

            input_size = img_meta['input_size']
            results.bboxes[:, 0::2].clamp_(0, input_size[0])
            results.bboxes[:, 1::2].clamp_(0, input_size[1])

            results_list.append(results.numpy())

        return results_list

    def switch_to_deploy(self, test_cfg: Optional[Dict]):
        """Precompute and save the grid coordinates and strides."""

        if getattr(self, 'deploy', False):
            return

        self.deploy = True

        # grid generator
        input_size = test_cfg.get('input_size', (640, 640))
        featmaps = []
        for s in self.featmap_strides:
            featmaps.append(
                torch.rand(1, 1, input_size[0] // s, input_size[1] // s))
        featmap_sizes = [fmap.shape[2:] for fmap in featmaps]

        self.mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=torch.float32, device='cpu')
        self.flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            self.flatten_priors.new_full((featmap_size.numel(), ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        self.flatten_stride = torch.cat(mlvl_strides)

class AdaptiveRouter(nn.Module):
    def __init__(self, features_channels, out_channels, reduction=4):
        super(AdaptiveRouter, self).__init__()
        self.inp = sum(features_channels)
        self.oup = out_channels
        self.reduction = reduction
        self.layer1 = nn.Conv2d(self.inp, self.inp // self.reduction, kernel_size=1, bias=True)
        self.layer2 = nn.Conv2d(self.inp // self.reduction, self.oup, kernel_size=1, bias=True)

    def forward(self, xs):
        xs = [x.mean(dim=(2, 3), keepdim=True) for x in xs]
        xs = torch.cat(xs, dim=1)
        xs = self.layer1(xs)
        xs = F.relu(xs, inplace=True)
        xs = self.layer2(xs).flatten(1)
        xs = xs.sigmoid()
        # if self.training:
        #     xs = xs.sigmoid()
        #     # xs = sigmoid(xs, hard=False, threshold=thres)
        # else:
        #     xs = xs.sigmoid()
        return xs


