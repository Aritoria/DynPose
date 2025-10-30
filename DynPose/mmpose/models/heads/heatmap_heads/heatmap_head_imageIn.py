# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead
from mmpose.registry import MODELS
from .checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
                         find_latest_checkpoint, save_checkpoint,
                         weights_to_cpu)

import torch.nn.functional as F

OptIntSeq = Optional[Sequence[int]]
#
# @MODELS.register_module()
# class HeatmapHeadImageIn(BaseHead):
#     """Top-down heatmap head introduced in `Simple Baselines`_ by Xiao et al
#     (2018). The head is composed of a few deconvolutional layers followed by a
#     convolutional layer to generate heatmaps from low-resolution feature maps.
#
#     Args:
#         in_channels (int | Sequence[int]): Number of channels in the input
#             feature map
#         out_channels (int): Number of channels in the output heatmap
#         deconv_out_channels (Sequence[int], optional): The output channel
#             number of each deconv layer. Defaults to ``(256, 256, 256)``
#         deconv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
#             of each deconv layer. Each element should be either an integer for
#             both height and width dimensions, or a tuple of two integers for
#             the height and the width dimension respectively.Defaults to
#             ``(4, 4, 4)``
#         conv_out_channels (Sequence[int], optional): The output channel number
#             of each intermediate conv layer. ``None`` means no intermediate
#             conv layer between deconv layers and the final conv layer.
#             Defaults to ``None``
#         conv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
#             of each intermediate conv layer. Defaults to ``None``
#         final_layer (dict): Arguments of the final Conv2d layer.
#             Defaults to ``dict(kernel_size=1)``
#         loss (Config): Config of the keypoint loss. Defaults to use
#             :class:`KeypointMSELoss`
#         decoder (Config, optional): The decoder config that controls decoding
#             keypoint coordinates from the network output. Defaults to ``None``
#         init_cfg (Config, optional): Config to control the initialization. See
#             :attr:`default_init_cfg` for default settings
#
#     .. _`Simple Baselines`: https://arxiv.org/abs/1804.06208
#     """
#
#     _version = 2
#
#     def __init__(self,
#                  in_channels: Union[int, Sequence[int]],
#                  out_channels: int,
#                  deconv_out_channels: OptIntSeq = (256, 256, 256),
#                  deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
#                  conv_out_channels: OptIntSeq = None,
#                  conv_kernel_sizes: OptIntSeq = None,
#                  final_layer: dict = dict(kernel_size=1),
#                  loss: ConfigType = dict(
#                      type='KeypointMSELoss', use_target_weight=True),
#                  decoder: OptConfigType = None,
#                  init_cfg: OptConfigType = None):
#
#         if init_cfg is None:
#             init_cfg = self.default_init_cfg
#
#         super().__init__(init_cfg)
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.loss_module = MODELS.build(loss)
#         if decoder is not None:
#             self.decoder = KEYPOINT_CODECS.build(decoder)
#         else:
#             self.decoder = None
#
#         if deconv_out_channels:
#             if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
#                     deconv_kernel_sizes):
#                 raise ValueError(
#                     '"deconv_out_channels" and "deconv_kernel_sizes" should '
#                     'be integer sequences with the same length. Got '
#                     f'mismatched lengths {deconv_out_channels} and '
#                     f'{deconv_kernel_sizes}')
#
#             self.deconv_layers = self._make_deconv_layers(
#                 in_channels=in_channels,
#                 layer_out_channels=deconv_out_channels,
#                 layer_kernel_sizes=deconv_kernel_sizes,
#             )
#             in_channels = deconv_out_channels[-1]
#         else:
#             self.deconv_layers = nn.Identity()
#
#         if conv_out_channels:
#             if conv_kernel_sizes is None or len(conv_out_channels) != len(
#                     conv_kernel_sizes):
#                 raise ValueError(
#                     '"conv_out_channels" and "conv_kernel_sizes" should '
#                     'be integer sequences with the same length. Got '
#                     f'mismatched lengths {conv_out_channels} and '
#                     f'{conv_kernel_sizes}')
#
#             self.conv_layers = self._make_conv_layers(
#                 in_channels=in_channels,
#                 layer_out_channels=conv_out_channels,
#                 layer_kernel_sizes=conv_kernel_sizes)
#             in_channels = conv_out_channels[-1]
#         else:
#             self.conv_layers = nn.Identity()
#
#         if final_layer is not None:
#             cfg = dict(
#                 type='Conv2d',
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=1)
#             cfg.update(final_layer)
#             self.final_layer = build_conv_layer(cfg)
#         else:
#             self.final_layer = nn.Identity()
#
#         # Register the hook to automatically convert old version state dicts
#         self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
#
#         network2 =  dict(
#             type='TopdownPoseEstimator',
#             backbone=dict(
#                 type='HRNet',
#                 in_channels=3,
#                 extra=dict(
#                     stage1=dict(
#                         num_modules=1,
#                         num_branches=1,
#                         block='BOTTLENECK',
#                         num_blocks=(4,),
#                         num_channels=(64,)),
#                     stage2=dict(
#                         num_modules=1,
#                         num_branches=2,
#                         block='BASIC',
#                         num_blocks=(4, 4),
#                         num_channels=(32, 64)),
#                     stage3=dict(
#                         num_modules=4,
#                         num_branches=3,
#                         block='BASIC',
#                         num_blocks=(4, 4, 4),
#                         num_channels=(32, 64, 128)),
#                     stage4=dict(
#                         num_modules=3,
#                         num_branches=4,
#                         block='BASIC',
#                         num_blocks=(4, 4, 4, 4),
#                         num_channels=(32, 64, 128, 256)))
#             ),
#             head=dict(
#                 type='HeatmapHead',
#                 in_channels=32,
#                 out_channels=17,
#                 deconv_out_channels=None
#                 # deconv_out_channels=None,
#                 # loss=dict(type='KeypointMSELoss', use_target_weight=True)
#             ),
#         )
#         self.net2 = MODELS.build(network2)
#         self.is_load = 0
#
#         self.router = AdaptiveRouter([2048], 2, reduction=8)
#
#     def _make_conv_layers(self, in_channels: int,
#                           layer_out_channels: Sequence[int],
#                           layer_kernel_sizes: Sequence[int]) -> nn.Module:
#         """Create convolutional layers by given parameters."""
#
#         layers = []
#         for out_channels, kernel_size in zip(layer_out_channels,
#                                              layer_kernel_sizes):
#             padding = (kernel_size - 1) // 2
#             cfg = dict(
#                 type='Conv2d',
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 padding=padding)
#             layers.append(build_conv_layer(cfg))
#             layers.append(nn.BatchNorm2d(num_features=out_channels))
#             layers.append(nn.ReLU(inplace=True))
#             in_channels = out_channels
#
#         return nn.Sequential(*layers)
#
#     def _make_deconv_layers(self, in_channels: int,
#                             layer_out_channels: Sequence[int],
#                             layer_kernel_sizes: Sequence[int]) -> nn.Module:
#         """Create deconvolutional layers by given parameters."""
#
#         layers = []
#         for out_channels, kernel_size in zip(layer_out_channels,
#                                              layer_kernel_sizes):
#             if kernel_size == 4:
#                 padding = 1
#                 output_padding = 0
#             elif kernel_size == 3:
#                 padding = 1
#                 output_padding = 1
#             elif kernel_size == 2:
#                 padding = 0
#                 output_padding = 0
#             else:
#                 raise ValueError(f'Unsupported kernel size {kernel_size} for'
#                                  'deconvlutional layers in '
#                                  f'{self.__class__.__name__}')
#             cfg = dict(
#                 type='deconv',
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 stride=2,
#                 padding=padding,
#                 output_padding=output_padding,
#                 bias=False)
#             layers.append(build_upsample_layer(cfg))
#             layers.append(nn.BatchNorm2d(num_features=out_channels))
#             layers.append(nn.ReLU(inplace=True))
#             in_channels = out_channels
#
#         return nn.Sequential(*layers)
#
#     @property
#     def default_init_cfg(self):
#         init_cfg = [
#             dict(
#                 type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
#             dict(type='Constant', layer='BatchNorm2d', val=1)
#         ]
#         return init_cfg
#
#     def forward(self, feats: tuple[Tensor]) -> Tensor:
#         """Forward the network. The input is multi scale feature maps and the
#         output is the heatmap.
#
#         Args:
#             feats (Tuple[Tensor]): Multi scale feature maps.
#
#         Returns:
#             Tensor: output heatmap.
#         """
#
#         if self.is_load == 0:
#             self.load_checkpoint("td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth", 'cpu', False)
#             self.is_load = 1
#         # image = feats[-1]
#         feat_and_image = feats
#         image = feat_and_image[1]
#         x = feat_and_image[0]
#         # print(x.shape)
#
#         router_result = self.router([x]) # [bs,1]
#         # if router_result[0, 0] <= 0.5 or router_result[0, 1] <= 0.5:
#         #     print("Net 1")
#         # else:
#         #     print("Net 2")
#
#
#         if self.training:
#             with torch.no_grad():
#                 x = self.deconv_layers(x)
#                 x = self.conv_layers(x)
#                 x = self.final_layer(x)
#                 x_net2 = self.net2.backbone(image)
#                 x_net2 = self.net2.head(x_net2)
#             return x, x_net2, router_result
#         else:
#             if router_result[0,0] <= 0.5 or router_result[0,1] <= 0.5:
#                 x = self.deconv_layers(x)
#                 x = self.conv_layers(x)
#                 x = self.final_layer(x)
#                 return x
#             else :
#                 x_net2 = self.net2.backbone(image)
#                 x_net2 = self.net2.head(x_net2)
#                 return x_net2
#
#
#         # x = self.deconv_layers(x)
#         # x = self.conv_layers(x)
#         # x = self.final_layer(x)
#         # return x
#
#
#
#     def predict(self,
#                 feats: Features,
#                 batch_data_samples: OptSampleList,
#                 test_cfg: ConfigType = {}) -> Predictions:
#         """Predict results from features.
#
#         Args:
#             feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
#                 features (or multiple multi-stage features in TTA)
#             batch_data_samples (List[:obj:`PoseDataSample`]): The batch
#                 data samples
#             test_cfg (dict): The runtime config for testing process. Defaults
#                 to {}
#
#         Returns:
#             Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
#             ``test_cfg['output_heatmap']==True``, return both pose and heatmap
#             prediction; otherwise only return the pose prediction.
#
#             The pose prediction is a list of ``InstanceData``, each contains
#             the following fields:
#
#                 - keypoints (np.ndarray): predicted keypoint coordinates in
#                     shape (num_instances, K, D) where K is the keypoint number
#                     and D is the keypoint dimension
#                 - keypoint_scores (np.ndarray): predicted keypoint scores in
#                     shape (num_instances, K)
#
#             The heatmap prediction is a list of ``PixelData``, each contains
#             the following fields:
#
#                 - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
#         """
#
#         if test_cfg.get('flip_test', False):
#             # TTA: flip test -> feats = [orig, flipped]
#             assert isinstance(feats, list) and len(feats) == 2
#             flip_indices = batch_data_samples[0].metainfo['flip_indices']
#             _feats, _feats_flip = feats
#             _batch_heatmaps = self.forward(_feats)
#             _batch_heatmaps_flip = flip_heatmaps(
#                 self.forward(_feats_flip),
#                 flip_mode=test_cfg.get('flip_mode', 'heatmap'),
#                 flip_indices=flip_indices,
#                 shift_heatmap=test_cfg.get('shift_heatmap', False))
#             batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
#         else:
#             batch_heatmaps = self.forward(feats)
#
#         preds = self.decode(batch_heatmaps)
#
#         if test_cfg.get('output_heatmaps', False):
#             pred_fields = [
#                 PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
#             ]
#             return preds, pred_fields
#         else:
#             return preds
#
#     def loss(self,
#              feats: Tuple[Tensor],
#              batch_data_samples: OptSampleList,
#              train_cfg: ConfigType = {}) -> dict:
#         """Calculate losses from a batch of inputs and data samples.
#
#         Args:
#             feats (Tuple[Tensor]): The multi-stage features
#             batch_data_samples (List[:obj:`PoseDataSample`]): The batch
#                 data samples
#             train_cfg (dict): The runtime config for training process.
#                 Defaults to {}
#
#         Returns:
#             dict: A dictionary of losses.
#         """
#         # pred_fields = self.forward(feats)
#         heatmap_head1, heatmap_head2, score = self.forward(feats)
#         gt_heatmaps = torch.stack(
#             [d.gt_fields.heatmaps for d in batch_data_samples])
#         keypoint_weights = torch.cat([
#             d.gt_instance_labels.keypoint_weights for d in batch_data_samples
#         ])
#
#         # calculate losses
#         losses = dict()
#         # loss = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)
#         loss_head1 = self.loss_module(heatmap_head1, gt_heatmaps, keypoint_weights) * 1000
#         loss_head2 = self.loss_module(heatmap_head2, gt_heatmaps, keypoint_weights) * 1000
#
#         delt = loss_head1-loss_head2
#         # print("delt",delt,"score",score)
#
#         loss1 = (1.0 - score[0,0])*(loss_head1-(delt/2.0)) + score[0,0]*(loss_head2+(delt/2.0))
#         loss2 = (score[0,1] - delt)**2
#
#         losses.update(loss_kpt=loss1 + loss2)
#
#         pred_fields = heatmap_head2.clone()
#         # indices_gt_05 = torch.where(score[:,0]>0.5)[0]
#         # pred_fields[indices_gt_05] = heatmap_head2[indices_gt_05]
#
#         # calculate accuracy
#         if train_cfg.get('compute_acc', True):
#             _, avg_acc, _ = pose_pck_accuracy(
#                 output=to_numpy(pred_fields),
#                 target=to_numpy(gt_heatmaps),
#                 mask=to_numpy(keypoint_weights) > 0)
#
#             acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
#             losses.update(acc_pose=acc_pose)
#
#         return losses
#
#     def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
#                                   **kwargs):
#         """A hook function to convert old-version state dict of
#         :class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
#         compatible format of :class:`HeatmapHead`.
#
#         The hook will be automatically registered during initialization.
#         """
#         version = local_meta.get('version', None)
#         if version and version >= self._version:
#             return
#
#         # convert old-version state dict
#         keys = list(state_dict.keys())
#         for _k in keys:
#             if not _k.startswith(prefix):
#                 continue
#             v = state_dict.pop(_k)
#             k = _k[len(prefix):]
#             # In old version, "final_layer" includes both intermediate
#             # conv layers (new "conv_layers") and final conv layers (new
#             # "final_layer").
#             #
#             # If there is no intermediate conv layer, old "final_layer" will
#             # have keys like "final_layer.xxx", which should be still
#             # named "final_layer.xxx";
#             #
#             # If there are intermediate conv layers, old "final_layer"  will
#             # have keys like "final_layer.n.xxx", where the weights of the last
#             # one should be renamed "final_layer.xxx", and others should be
#             # renamed "conv_layers.n.xxx"
#             k_parts = k.split('.')
#             if k_parts[0] == 'final_layer':
#                 if len(k_parts) == 3:
#                     assert isinstance(self.conv_layers, nn.Sequential)
#                     idx = int(k_parts[1])
#                     if idx < len(self.conv_layers):
#                         # final_layer.n.xxx -> conv_layers.n.xxx
#                         k_new = 'conv_layers.' + '.'.join(k_parts[1:])
#                     else:
#                         # final_layer.n.xxx -> final_layer.xxx
#                         k_new = 'final_layer.' + k_parts[2]
#                 else:
#                     # final_layer.xxx remains final_layer.xxx
#                     k_new = k
#             else:
#                 k_new = k
#
#             state_dict[prefix + k_new] = v
#
#     def load_checkpoint(self,
#                         filename: str,
#                         map_location,
#                         strict,
#                         revise_keys: list = [(r'^module.', '')]):
#         """Load checkpoint from given ``filename``.
#
#         Args:
#             filename (str): Accept local filepath, URL, ``torchvision://xxx``,
#                 ``open-mmlab://xxx``.
#             map_location (str or callable): A string or a callable function to
#                 specifying how to remap storage locations.
#                 Defaults to 'cpu'.
#             strict (bool): strict (bool): Whether to allow different params for
#                 the model and checkpoint.
#             revise_keys (list): A list of customized keywords to modify the
#                 state_dict in checkpoint. Each item is a (pattern, replacement)
#                 pair of the regular expression operations. Defaults to strip
#                 the prefix 'module.' by [(r'^module\\.', '')].
#         """
#         checkpoint = _load_checkpoint(filename, map_location=map_location)
#         model = self.net2
#         checkpoint = _load_checkpoint_to_model(
#             model, checkpoint, strict, revise_keys=revise_keys)
#         print("Loaded HRNet checkpoint")
#
#         return checkpoint
#
#
# class AdaptiveRouter(nn.Module):
#     def __init__(self, features_channels, out_channels, reduction=4):
#         super(AdaptiveRouter, self).__init__()
#         self.inp = sum(features_channels)
#         self.oup = out_channels
#         self.reduction = reduction
#         self.layer1 = nn.Conv2d(self.inp, self.inp // self.reduction, kernel_size=1, bias=True)
#         self.layer2 = nn.Conv2d(self.inp // self.reduction, self.oup, kernel_size=1, bias=True)
#
#     def forward(self, xs):
#         xs = [x.mean(dim=(2, 3), keepdim=True) for x in xs]
#         xs = torch.cat(xs, dim=1)
#         xs = self.layer1(xs)
#         xs = F.relu(xs, inplace=True)
#         xs = self.layer2(xs).flatten(1)
#         xs = xs.sigmoid()
#         # if self.training:
#         #     xs = xs.sigmoid()
#         #     # xs = sigmoid(xs, hard=False, threshold=thres)
#         # else:
#         #     xs = xs.sigmoid()
#         return xs
#
#
# # def sigmoid(logits, hard=False, threshold=0.5):
# #     y_soft = logits.sigmoid()
# #     if hard:
# #         indices = (y_soft < threshold).nonzero(as_tuple=True)
# #         y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
# #         y_hard[indices[0], indices[1]] = 1.0
# #         ret = y_hard - y_soft.detach() + y_soft
# #     else:
# #         ret = y_soft
# #     return ret
#











@MODELS.register_module()
class HeatmapHeadImageIn(BaseHead):
    """Top-down heatmap head introduced in `Simple Baselines`_ by Xiao et al
    (2018). The head is composed of a few deconvolutional layers followed by a
    convolutional layer to generate heatmaps from low-resolution feature maps.

    Args:
        in_channels (int | Sequence[int]): Number of channels in the input
            feature map
        out_channels (int): Number of channels in the output heatmap
        deconv_out_channels (Sequence[int], optional): The output channel
            number of each deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        conv_out_channels (Sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        final_layer (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KeypointMSELoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`Simple Baselines`: https://arxiv.org/abs/1804.06208
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 final_layer: dict = dict(kernel_size=1),
                 loss: ConfigType = dict(
                     type='KeypointMSELoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                    conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {conv_out_channels} and '
                    f'{conv_kernel_sizes}')

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        if final_layer is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            cfg.update(final_layer)
            self.final_layer = build_conv_layer(cfg)
        else:
            self.final_layer = nn.Identity()

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

        network2 =  dict(
            type='TopdownPoseEstimator',
            backbone=dict(
                type='HRNet',
                frozen_stages=4,
                in_channels=3,
                extra=dict(
                    stage1=dict(
                        num_modules=1,
                        num_branches=1,
                        block='BOTTLENECK',
                        num_blocks=(4,),
                        num_channels=(64,)),
                    stage2=dict(
                        num_modules=1,
                        num_branches=2,
                        block='BASIC',
                        num_blocks=(4, 4),
                        num_channels=(32, 64)),
                    stage3=dict(
                        num_modules=4,
                        num_branches=3,
                        block='BASIC',
                        num_blocks=(4, 4, 4),
                        num_channels=(32, 64, 128)),
                    stage4=dict(
                        num_modules=3,
                        num_branches=4,
                        block='BASIC',
                        num_blocks=(4, 4, 4, 4),
                        num_channels=(32, 64, 128, 256)))
            ),
            head=dict(
                type='HeatmapHead',
                in_channels=32,
                out_channels=17,
                deconv_out_channels=None
                # deconv_out_channels=None,
                # loss=dict(type='KeypointMSELoss', use_target_weight=True)
            ),
        )
        self.net2 = MODELS.build(network2)
        for k, v in self.net2.named_parameters():
            if v.requires_grad == True:
                v.requires_grad = False
        for k, v in self.deconv_layers.named_parameters():
            if v.requires_grad == True:
                v.requires_grad = False
        for k, v in self.conv_layers.named_parameters():
            if v.requires_grad == True:
                v.requires_grad = False
        for k, v in self.final_layer.named_parameters():
            if v.requires_grad == True:
                v.requires_grad = False
        self.is_load = 0
        self.test_load = 0

        self.router = AdaptiveRouter([2048], 1, reduction=4)

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
            layers.append(build_conv_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(build_upsample_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def forward(self, feats: tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """

        if self.is_load == 0:
            self.load_checkpoint("td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth", 'cpu', False)
            self.is_load = 1

        if self.training:
            pass
        elif self.test_load == 0:
            self.load_checkpoint("td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth", 'cpu', False)
            self.test_load = 1
        else:
            pass

        # image = feats[-1]
        feat_and_image = feats
        image = feat_and_image[1]
        x = feat_and_image[0]
        # print(x.shape)

        router_result = self.router([x]) # [bs,1]
        # if router_result[0, 0] <= 0.5:
        #     print("Net 1 ",router_result)
        # else:
        #     print("Net 2",router_result)
        # print(router_result)

        if self.training:
            with torch.no_grad():
                x = self.deconv_layers(x)
                x = self.conv_layers(x)
                x = self.final_layer(x)
                x_net2 = self.net2.backbone(image)
                x_net2 = self.net2.head(x_net2)

            return x, x_net2, router_result
        else:
            # router_result[0,0] = 1
            # print(router_result[0,0])
            if router_result[0,0] <= 0.49909:
                # print("net1")
                with torch.no_grad():
                    x = self.deconv_layers(x)
                    x = self.conv_layers(x)
                    x = self.final_layer(x)
                return x
            else :
                # print("net2")
                with torch.no_grad():
                    x_net2 = self.net2.backbone(image)
                    x_net2 = self.net2.head(x_net2)
                return x_net2
        # x = self.deconv_layers(x)
        # x = self.conv_layers(x)
        # x = self.final_layer(x)
        #
        # return x
        # return x

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
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

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _batch_heatmaps = self.forward(_feats)
            _batch_heatmaps_flip = flip_heatmaps(
                self.forward(_feats_flip),
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            batch_heatmaps = self.forward(feats)

        preds = self.decode(batch_heatmaps)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields
        else:
            return preds

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
        # pred_fields = self.forward(feats)
        heatmap_head1, heatmap_head2, score = self.forward(feats)
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # calculate losses
        losses = dict()
        # loss = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)
        loss_head1 = self.loss_module(heatmap_head1, gt_heatmaps, keypoint_weights) * 1000
        loss_head2 = self.loss_module(heatmap_head2, gt_heatmaps, keypoint_weights) * 1000

        delt = loss_head1-loss_head2
        # print("delt",delt,"score",score)

        loss = (1.0 - score[0,0])*(loss_head1-(delt/2.0)) + score[0,0]*(loss_head2+(delt/2.0))

        losses.update(loss_kpt=loss)

        pred_fields = heatmap_head2.clone()
        # indices_gt_05 = torch.where(score[:,0]>0.5)[0]
        # pred_fields[indices_gt_05] = heatmap_head2[indices_gt_05]

        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)

            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
        compatible format of :class:`HeatmapHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            k = _k[len(prefix):]
            # In old version, "final_layer" includes both intermediate
            # conv layers (new "conv_layers") and final conv layers (new
            # "final_layer").
            #
            # If there is no intermediate conv layer, old "final_layer" will
            # have keys like "final_layer.xxx", which should be still
            # named "final_layer.xxx";
            #
            # If there are intermediate conv layers, old "final_layer"  will
            # have keys like "final_layer.n.xxx", where the weights of the last
            # one should be renamed "final_layer.xxx", and others should be
            # renamed "conv_layers.n.xxx"
            k_parts = k.split('.')
            if k_parts[0] == 'final_layer':
                if len(k_parts) == 3:
                    assert isinstance(self.conv_layers, nn.Sequential)
                    idx = int(k_parts[1])
                    if idx < len(self.conv_layers):
                        # final_layer.n.xxx -> conv_layers.n.xxx
                        k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                    else:
                        # final_layer.n.xxx -> final_layer.xxx
                        k_new = 'final_layer.' + k_parts[2]
                else:
                    # final_layer.xxx remains final_layer.xxx
                    k_new = k
            else:
                k_new = k

            state_dict[prefix + k_new] = v

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
        checkpoint = _load_checkpoint_to_model(model, checkpoint, strict, revise_keys=revise_keys)
        print("Loaded HRNet checkpoint")

        return checkpoint


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


# def sigmoid(logits, hard=False, threshold=0.5):
#     y_soft = logits.sigmoid()
#     if hard:
#         indices = (y_soft < threshold).nonzero(as_tuple=True)
#         y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
#         y_hard[indices[0], indices[1]] = 1.0
#         ret = y_hard - y_soft.detach() + y_soft
#     else:
#         ret = y_soft
#     return ret
