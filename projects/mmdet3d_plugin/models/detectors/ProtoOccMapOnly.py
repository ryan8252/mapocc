# Copyright (c) Phigent Robotics. All rights reserved.
import numpy as np
import torch

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
                 **kwargs):
        super(ProtoOccMapOnly, self).__init__(**kwargs)
        self.pts_bbox_head = None

        self.depth_net = build_head(depth_net)
        self.map_bev_encoder = builder.build_backbone(map_bev_encoder)
        self.bev_seg_head = build_head(bev_seg_head)
        self.map_loss_weight = map_loss_weight
        self.train_depth = train_depth

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

    def _check_map_feature_size(self, map_feature, gt_masks_bev):
        if gt_masks_bev is None:
            return
        if map_feature.shape[-2:] != gt_masks_bev.shape[-2:]:
            raise ValueError(
                f'map feature size {map_feature.shape[-2:]} does not match '
                f'gt_masks_bev size {gt_masks_bev.shape[-2:]}.')

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
