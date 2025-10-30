# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn
torch.set_printoptions(profile="full")
torch.set_printoptions(precision=10)
from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead
from .checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
                         find_latest_checkpoint, save_checkpoint,
                         weights_to_cpu)

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class TwoHead(BaseHead):
    _version = 2

    def __init__(self, loss: ConfigType = dict(
        type='KeypointMSELoss', use_target_weight=True),
                 decoder: OptConfigType = None, ):
        super().__init__()

        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        network1 = dict(
            type='TopdownPoseEstimator',
            backbone=dict(
                type='ResNet',
                depth=50,
                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
            ),
            head=dict(
                type='HeatmapHead',
                in_channels=2048,
                out_channels=17,
            ),
        )


        network2 = dict(
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
            ),
        )


        self.net2 = MODELS.build(network2)
        self.is_load = 0
        self.net1_num = 0
        self.net2_num = 0

        self.l1 = []
        self.l2 = []

        # Register the hook to automatically convert old version state dicts
        # self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def forward(self, feats: Tuple[Tensor], flag=0) -> Tensor:

        if self.is_load == 0:
            self.load_checkpoint('cpu', False)
            self.is_load = 1

        image = feats[0]
        scores = feats[1]
        # print("I'm in TwoHead")

        if self.training:
            with torch.no_grad():
                x_net1 = self.net1.backbone(image)
                x_net1 = self.net1.head(x_net1)
                x_net2 = self.net2.backbone(image)
                x_net2 = self.net2.head(x_net2)
            # print("x",x.shape)
            return x_net1, x_net2, scores
        else:

            b = scores.size(0)
            if flag == 0:
                print('net1_num:', self.net1_num, '   net2_num:', self.net2_num)

                self.indices_net2 = torch.nonzero((scores[:, 1] >= 0.59) & (scores[:, 1] <= 0.69), as_tuple=False).flatten()
                self.indices_net1 = torch.nonzero((scores[:, 1] < 0.59) | (scores[:, 1] > 0.69), as_tuple=False).flatten()


                result = torch.zeros((b, 17, 64, 48), device=scores.device).contiguous()

                self.net1_num = self.net1_num + len(self.indices_net1)
                self.net2_num = self.net2_num + len(self.indices_net2)
                # print(self.indices_net2, self.indices_net1)

                if self.indices_net2.size(0) != 0:
                    result[self.indices_net2] = self.net2.head(self.net2.backbone(image[self.indices_net2]))
                if self.indices_net1.size(0) != 0:
                    result[self.indices_net1] = self.net1.head(self.net1.backbone(image[self.indices_net1]))
                # print("First step of para flip is OK", result.shape)
                return result , 1
            else:
                result = torch.zeros((b, 17, 64, 48), device=scores.device)
                if self.indices_net2.size(0) != 0:
                    result[self.indices_net2] = self.net2.head(self.net2.backbone(image[self.indices_net2]))
                if self.indices_net1.size(0) != 0:
                    result[self.indices_net1] = self.net1.head(self.net1.backbone(image[self.indices_net1]))
                # print("Sec step of para flip is OK", result.shape)
                self.indices_net1 = None
                self.indices_net2 = None
                return result.contiguous() , 2




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
            _batch_heatmaps, flag = self.forward(_feats, 0)
            new_feature, flag = self.forward(_feats_flip, flag)
            _batch_heatmaps_flip = flip_heatmaps(
                new_feature,
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            batch_heatmaps = self.forward(feats)


        preds = self.decode(batch_heatmaps)


        torch.cuda.empty_cache()

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
        pred_fields_1, pred_fields_2, scores = self.forward(feats)
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_labels = torch.cat([
            d.gt_instance_labels.keypoint_labels for d in batch_data_samples
        ])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        preds_1 = torch.tensor(self.decode(pred_fields_1)[0]['keypoints'], device=pred_fields_1.device)
        preds_2 = torch.tensor(self.decode(pred_fields_2)[0]['keypoints'], device=pred_fields_1.device)

        OKS1 = computeOks(preds_1.reshape(-1),
                          (keypoint_labels * torch.tensor([192, 256], device=pred_fields_1.device)).reshape(-1),
                          keypoint_weights.reshape(-1))
        OKS2 = computeOks(preds_2.reshape(-1),
                          (keypoint_labels * torch.tensor([192, 256], device=pred_fields_1.device)).reshape(-1),
                          keypoint_weights.reshape(-1))

        losses = dict()
        dis_1np = self.pose_dist(
            output=to_numpy(pred_fields_1),
            target=to_numpy(gt_heatmaps),
            mask=to_numpy(keypoint_weights) > 0
        )
        dis_2np = self.pose_dist(
            output=to_numpy(pred_fields_2),
            target=to_numpy(gt_heatmaps),
            mask=to_numpy(keypoint_weights) > 0
        )
        # print('dis_1np', dis_1np.shape)
        sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])
        vars = (sigmas * 2) ** 2
        sigma1 = torch.tensor([1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 2., 2., 1.2, 1.2, 1.5, 1.5, 2., 2.])
        dis_1 = ((torch.from_numpy(dis_1np).flatten() / vars) * sigma1).sum() * 100
        dis_2 = ((torch.from_numpy(dis_2np).flatten() / vars) * sigma1).sum() * 100
        delt = dis_1 - dis_2
        if np.abs(OKS2 - OKS1) > 0.05 and np.abs(OKS2 - OKS1) < 0.3:
            loss_score1 = (1.0 - scores[0, 0]) * (dis_1 - (delt / 2.0)) + scores[0, 0] * (dis_2 + (delt / 2.0))
            if dis_1 < 0.001:
                loss_score2 = torch.tensor(0.0, device=pred_fields_1.device)
            else:
                loss_score2 = ((delt / dis_1) - scores[0, 1]) * ((delt / dis_1) - scores[0, 1])
        else:
            loss_score1 = torch.tensor(0., device=pred_fields_1.device)
            loss_score2 = torch.tensor(0., device=pred_fields_1.device)

        sig3 = 2.0

        loss_score3 = (scores[0, 2] - sig3 * (1 - OKS1)) * (scores[0, 2] - sig3 * (1 - OKS1))  # 预测人体检测结果的困难程度

        losses.update(loss_score1=loss_score1)
        losses.update(loss_score2=loss_score2)
        losses.update(loss_score3=loss_score3)
        return losses

    def pose_dist(self, output: np.ndarray,
                  target: np.ndarray,
                  mask: np.ndarray,
                  normalize: Optional[np.ndarray] = None) -> tuple:
        N, K, H, W = output.shape
        if K == 0:
            return None, 0, 0
        if normalize is None:
            normalize = np.tile(np.array([[H, W]]), (N, 1))

        pred, _ = get_heatmap_maximum(output)
        gt, _ = get_heatmap_maximum(target)
        distances = _calc_distances(pred, gt, mask, normalize)
        return distances

    def load_checkpoint(self,
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
        checkpoint1 = _load_checkpoint('td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth',
                                       map_location=map_location)
        checkpoint1 = _load_checkpoint_to_model(self.net2, checkpoint1, strict, revise_keys=revise_keys)
        print("Loaded HRNet checkpoint")

        checkpoint2 = _load_checkpoint('td-hm_res50_8xb64-210e_coco-256x192-04af38ce_20220923.pth',
                                       map_location=map_location)
        checkpoint2 = _load_checkpoint_to_model(self.net1, checkpoint2, strict, revise_keys=revise_keys)
        print("Loaded Resnet checkpoint")


        return checkpoint1, checkpoint2

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


def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, (
        f'Invalid shape {heatmaps.shape}')

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)

    return locs, vals


def _calc_distances(preds: np.ndarray, gts: np.ndarray, mask: np.ndarray,
                    norm_factor: np.ndarray) -> np.ndarray:
    N, K, _ = preds.shape
    # set mask=0 when norm_factor==0
    _mask = mask.copy()
    _mask[np.where((norm_factor == 0).sum(1))[0], :] = False

    distances = np.full((N, K), 0, dtype=np.float32)
    # handle invalid values
    norm_factor[np.where(norm_factor <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(((preds - gts) / norm_factor[:, None, :])[_mask], axis=-1)
    return distances.T


def computeOks(dt_pose, gt_pose, vi):
    dt_pose = dt_pose.cpu().numpy()
    gt_pose = gt_pose.cpu().numpy()
    vi = vi.cpu().numpy()

    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    area = 192 * 256
    vars = (sigmas * 2) ** 2
    k = len(sigmas)

    g = gt_pose
    xg = g[0::2]
    yg = g[1::2]
    vg = vi
    k1 = np.count_nonzero(vg > 0)
    bb = [0, 0, 192, 256]
    x0 = bb[0] - bb[2]
    x1 = bb[0] + bb[2] * 2
    y0 = bb[1] - bb[3]
    y1 = bb[1] + bb[3] * 2

    d = dt_pose
    xd = d[0::2]
    yd = d[1::2]
    if k1 > 0:
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg
    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        z = np.zeros((k))
        dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
        dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)

    e = (dx ** 2 + dy ** 2) / vars / (area + np.spacing(1)) / 2

    if k1 > 0:
        e = e[vg > 0]

    ious = np.sum(np.exp(-e)) / e.shape[0]

    return ious
