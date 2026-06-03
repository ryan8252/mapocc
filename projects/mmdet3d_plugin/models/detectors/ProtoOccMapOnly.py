# Copyright (c) Phigent Robotics. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F

from .bevdet import BEVDet
from mmdet3d.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.builder import build_head


@DETECTORS.register_module()
class ProtoOccMapOnly(BEVDet):
    """Map-only detector for BEV segmentation upper-bound diagnostics."""

    def __init__(self,
                 depth_net=None,
                 map_bev_encoder=None,
                 bev_seg_head=None,
                 map_loss_weight=1.0,
                 train_depth=True,
                 shared_feature_range=None,
                 map_feature_range=None,
                 map_feature_size=None,
                 **kwargs):
        super(ProtoOccMapOnly, self).__init__(**kwargs)
        self.pts_bbox_head = None

        self.depth_net = build_head(depth_net)
        self.map_bev_encoder = builder.build_backbone(map_bev_encoder)
        self.bev_seg_head = build_head(bev_seg_head)
        self.map_loss_weight = map_loss_weight
        self.train_depth = train_depth
        # Optional BEVFusion-aligned map grid: resample the map feature from the
        # shared feature range (e.g. +-51.2m) onto the map eval grid
        # (e.g. +-50m / 0.5m). Defaults keep the legacy single-grid behaviour.
        self.shared_feature_range = (
            torch.tensor(shared_feature_range, dtype=torch.float32)
            if shared_feature_range is not None else None)
        self.map_feature_range = (
            None if map_feature_range is None
            else torch.tensor(map_feature_range, dtype=torch.float32))
        self.map_feature_size = (
            None if map_feature_size is None
            else tuple(int(v) for v in map_feature_size))
        if self.map_feature_range is not None and self.shared_feature_range is None:
            raise ValueError(
                'map_feature_range requires shared_feature_range to resample '
                'the map feature onto the map eval grid.')

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, output_H, output_W = x.shape
        x = x.view(B, N, output_dim, output_H, output_W)
        return x, stereo_feat

    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        img_inputs = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(img_inputs[0])
        cam_params = img_inputs[1:7]

        mlp_input = self.depth_net.get_mlp_input(*cam_params)
        pv_feat, depth = self.depth_net(x, mlp_input)
        voxel_feat, depth = self.img_view_transformer(
            depth, pv_feat, [x] + img_inputs[1:7])
        return voxel_feat, depth, pv_feat

    def extract_feat(self, img_inputs, img_metas, **kwargs):
        return self.extract_img_feat(img_inputs, img_metas, **kwargs)

    def _scale_map_losses(self, map_losses):
        if self.map_loss_weight == 1.0:
            return map_losses
        return {name: loss * self.map_loss_weight
                for name, loss in map_losses.items()}

    def _normalize_map_targets(self, gt_masks_bev):
        if gt_masks_bev is None:
            return None
        while isinstance(gt_masks_bev, (list, tuple)) and len(gt_masks_bev) == 1:
            gt_masks_bev = gt_masks_bev[0]
        if isinstance(gt_masks_bev, (list, tuple)):
            if all(torch.is_tensor(item) for item in gt_masks_bev):
                gt_masks_bev = torch.stack(list(gt_masks_bev), dim=0)
            else:
                return None
        if not torch.is_tensor(gt_masks_bev):
            return None
        if gt_masks_bev.dim() == 5 and gt_masks_bev.size(0) == 1:
            gt_masks_bev = gt_masks_bev[0]
        if gt_masks_bev.dim() == 3:
            gt_masks_bev = gt_masks_bev.unsqueeze(0)
        return gt_masks_bev.float()

    def _get_map_feature_tensor(self, map_feature):
        if isinstance(map_feature, dict):
            map_tensor = map_feature.get('bev_feature', None)
            if map_tensor is None:
                raise ValueError('map_feature dict must contain `bev_feature`.')
            return map_tensor
        return map_feature

    def _check_map_feature_size(self, map_feature, gt_masks_bev):
        if gt_masks_bev is None:
            return
        map_tensor = self._get_map_feature_tensor(map_feature)
        if map_tensor.shape[-2:] != gt_masks_bev.shape[-2:]:
            raise ValueError(
                f'map feature size {map_tensor.shape[-2:]} does not match '
                f'gt_masks_bev size {gt_masks_bev.shape[-2:]}.')

    def _bev_xy_range(self, value, device, dtype):
        value = value.to(device=device, dtype=dtype)
        if value.numel() == 6:
            return value[[0, 1, 3, 4]]
        if value.numel() == 4:
            return value
        raise ValueError(
            'Expected a BEV range with 4 values or an XYZ range with 6 values, '
            f'but got {value.numel()} values.')

    def _make_bev_grid(self, feature, source_range, target_range, target_size):
        target_x, target_y = target_size
        device = feature.device
        dtype = feature.dtype
        source = self._bev_xy_range(source_range, device, dtype)
        target = self._bev_xy_range(target_range, device, dtype)

        xs = (
            torch.arange(target_x, device=device, dtype=dtype) + 0.5
        ) * ((target[2] - target[0]) / target_x) + target[0]
        ys = (
            torch.arange(target_y, device=device, dtype=dtype) + 0.5
        ) * ((target[3] - target[1]) / target_y) + target[1]

        x_norm = 2.0 * (xs - source[0]) / (source[2] - source[0]) - 1.0
        y_norm = 2.0 * (ys - source[1]) / (source[3] - source[1]) - 1.0
        try:
            grid_x, grid_y = torch.meshgrid(x_norm, y_norm, indexing='ij')
        except TypeError:
            grid_x, grid_y = torch.meshgrid(x_norm, y_norm)
        # grid_sample treats dim 0 as image-y/H and dim 1 as image-x/W.
        # This project stores BEV tensors as [x_cells, y_cells], so swap.
        grid = torch.stack((grid_y, grid_x), dim=-1)
        return grid.unsqueeze(0).expand(feature.shape[0], -1, -1, -1)

    def _align_map_tensor(self, map_feature):
        if self.map_feature_range is None or self.map_feature_size is None:
            return map_feature
        if tuple(map_feature.shape[-2:]) == self.map_feature_size:
            source = self._bev_xy_range(
                self.shared_feature_range,
                map_feature.device,
                map_feature.dtype)
            target = self._bev_xy_range(
                self.map_feature_range,
                map_feature.device,
                map_feature.dtype)
            if torch.allclose(source, target):
                return map_feature
        grid = self._make_bev_grid(
            map_feature,
            self.shared_feature_range,
            self.map_feature_range,
            self.map_feature_size)
        return F.grid_sample(
            map_feature,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)

    def _align_map_aux_tensor(self, map_feature):
        if self.map_feature_range is None or self.map_feature_size is None:
            return map_feature
        grid = self._make_bev_grid(
            map_feature,
            self.shared_feature_range,
            self.map_feature_range,
            map_feature.shape[-2:])
        return F.grid_sample(
            map_feature,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)

    def _align_map_feature(self, map_feature):
        if not isinstance(map_feature, dict):
            return self._align_map_tensor(map_feature)

        aligned = dict(map_feature)
        aligned['bev_feature'] = self._align_map_tensor(
            self._get_map_feature_tensor(map_feature))
        aux_features = map_feature.get('aux_features', {}) or {}
        aligned['aux_features'] = {
            str(level): self._align_map_aux_tensor(feature)
            for level, feature in aux_features.items()
        }
        return aligned

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      sa_gt_depth=None,
                      sa_gt_semantic=None,
                      gt_masks_bev=None,
                      **kwargs):
        voxel_feat, depth, pv_feat = self.extract_feat(
            img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        map_feature = self.map_bev_encoder(voxel_feat)
        map_feature = self._align_map_feature(map_feature)

        losses = {}
        if self.train_depth:
            losses.update(self.depth_net.get_PV_loss(
                pv_feat, depth, sa_gt_depth, sa_gt_semantic))

        gt_masks_bev = self._normalize_map_targets(gt_masks_bev)
        if gt_masks_bev is None:
            raise ValueError('Expected `gt_masks_bev` when training ProtoOccMapOnly.')
        self._check_map_feature_size(map_feature, gt_masks_bev)
        bev_seg_logits = self.bev_seg_head(map_feature)
        losses.update(self._scale_map_losses(
            self.bev_seg_head.loss(bev_seg_logits, gt_masks_bev)))
        return losses

    def simple_test(self,
                    points=None,
                    img_metas=None,
                    img=None,
                    rescale=False,
                    **kwargs):
        voxel_feat, _, _ = self.extract_feat(
            img_inputs=img, img_metas=img_metas, **kwargs)
        map_feature = self.map_bev_encoder(voxel_feat)
        map_feature = self._align_map_feature(map_feature)
        map_probs = self.bev_seg_head.predict(
            self.bev_seg_head(map_feature)).detach().cpu().numpy()

        gt_masks_bev = self._normalize_map_targets(kwargs.get('gt_masks_bev'))
        if gt_masks_bev is not None:
            self._check_map_feature_size(map_feature, gt_masks_bev)
            gt_masks_bev = gt_masks_bev.detach().cpu().numpy()

        results = []
        for batch_index in range(map_probs.shape[0]):
            sample_result = {
                'masks_bev': map_probs[batch_index].astype(np.float32),
            }
            if gt_masks_bev is not None:
                sample_result['gt_masks_bev'] = gt_masks_bev[batch_index].astype(np.int64)
            results.append(sample_result)
        return results

    def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        return None
