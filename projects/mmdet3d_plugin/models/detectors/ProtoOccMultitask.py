# Copyright (c) Phigent Robotics. All rights reserved.
import numpy as np

from .bevdet import BEVDet
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
import torch
from mmdet3d.models import builder


@DETECTORS.register_module()
class ProtoOccMultitask(BEVDet):
    def __init__(self,
                 pc_range=[-40.0, -40.0, -1, 40.0, 40.0, 5.4],
                 grid_size=[200, 200, 16],
                 depth_net=None,
                 dual_branch_encoder=None,
                 cnn3d_decoder=None,
                 prototype_query_decoder=None,
                 proto_map_head=None,
                 map_loss_weight=1.0,
                 **kwargs):
        super(ProtoOccMultitask, self).__init__(**kwargs)
        self.pts_bbox_head = None  # useless
        self.pc_range  = torch.tensor(pc_range)
        self.grid_size = torch.tensor(grid_size)

        self.depth_net                = build_head(depth_net)
        self.dual_branch_encoder      = builder.build_backbone(dual_branch_encoder)
        self.cnn3d_decoder            = build_head(cnn3d_decoder)
        self.prototype_query_decoder  = build_head(prototype_query_decoder)
        self.proto_map_head           = build_head(proto_map_head) if proto_map_head is not None else None
        self.map_loss_weight          = map_loss_weight

    # ──────────────────────────────────────────────────────────────────────
    # Shared feature extraction (unchanged from ProtoOccCnnSegHead)
    # ──────────────────────────────────────────────────────────────────────

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
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        img_inputs = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(img_inputs[0])
        cam_params = img_inputs[1:7]
        mlp_input = self.depth_net.get_mlp_input(*cam_params)
        pv_feat, depth = self.depth_net(x, mlp_input)
        voxel_feat, depth = self.img_view_transformer(depth, pv_feat, [x] + img_inputs[1:7])
        return voxel_feat, depth, pv_feat

    def extract_feat(self, img_inputs, img_metas, **kwargs):
        return self.extract_img_feat(img_inputs, img_metas, **kwargs)

    def _split_encoder_output(self, encoder_output):
        if isinstance(encoder_output, tuple):
            if len(encoder_output) == 3:
                return encoder_output
            if len(encoder_output) == 2:
                comprehensive_voxel_feature, bev_feature = encoder_output
                return comprehensive_voxel_feature, bev_feature, None
            raise ValueError(
                'Expected Dual_Branch_Encoder to return 1, 2, or 3 outputs, '
                f'but got {len(encoder_output)}.')
        return encoder_output, None, None

    def _select_map_feature(self, bev_feature, map_bev_feature):
        map_feature = map_bev_feature if map_bev_feature is not None else bev_feature
        if map_feature is None:
            raise ValueError(
                'proto_map_head requires either bev_feature or map_bev_feature '
                'from Dual_Branch_Encoder.')
        return map_feature

    def _check_map_feature_size(self, map_feature, gt_masks_bev):
        if gt_masks_bev is None:
            return
        if map_feature.shape[-2:] != gt_masks_bev.shape[-2:]:
            raise ValueError(
                f'map feature size {map_feature.shape[-2:]} does not match '
                f'gt_masks_bev size {gt_masks_bev.shape[-2:]}.')

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

    # ──────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      sa_gt_depth=None,
                      sa_gt_semantic=None,
                      gt_masks_bev=None,
                      non_vis_semantic_voxel=None,
                      **kwargs):
        # 2D-to-3D view transformation
        voxel_feat, depth, pv_feat = self.extract_feat(
            img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        # Dual Branch Encoder
        encoder_output = self.dual_branch_encoder(voxel_feat)
        comprehensive_voxel_feature, bev_feature, map_bev_feature = \
            self._split_encoder_output(encoder_output)

        # 3D CNN → coarse occ prediction + mask feature
        prototoype_occ_pred, mask_feat = self.cnn3d_decoder(
            comprehensive_voxel_feature.permute(0, 1, 4, 2, 3))

        # Prototype Query Decoder (occupancy)
        B = comprehensive_voxel_feature.shape[0]
        img_metas_occ = [{'pc_range': self.pc_range, 'occ_size': self.grid_size}
                         for _ in range(B)]
        losses = self.prototype_query_decoder.forward_train(
            comprehensive_voxel_feature, img_metas_occ,
            voxel_semantics, mask_camera, mask_feat, prototoype_occ_pred)

        losses.update(self.cnn3d_decoder.loss(prototoype_occ_pred, voxel_semantics, mask_camera))
        losses.update(self.depth_net.get_PV_loss(pv_feat, depth, sa_gt_depth, sa_gt_semantic))

        # ── Prototype Map Head ─────────────────────────────────────────────
        if self.proto_map_head is not None:
            gt_masks_bev = self._normalize_map_targets(gt_masks_bev)
            if gt_masks_bev is None:
                raise ValueError('Expected gt_masks_bev when training ProtoOccMultitask with proto_map_head.')
            map_feature = self._select_map_feature(bev_feature, map_bev_feature)
            self._check_map_feature_size(map_feature, gt_masks_bev)
            coarse_pred, final_masks = self.proto_map_head(
                map_feature, gt_masks_bev=gt_masks_bev)
            map_losses = self.proto_map_head.loss(coarse_pred, final_masks, gt_masks_bev)
            losses.update(self._scale_map_losses(map_losses))

        return losses

    # ──────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # 2D-to-3D view transformation
        voxel_feat, depth, pv_feat = self.extract_feat(
            img_inputs=img, img_metas=img_metas, **kwargs)

        # Dual Branch Encoder
        encoder_output = self.dual_branch_encoder(voxel_feat)
        comprehensive_voxel_feature, bev_feature, map_bev_feature = \
            self._split_encoder_output(encoder_output)

        # 3D CNN + Prototype Query Decoder (occupancy)
        prototoype_occ_pred, mask_feat = self.cnn3d_decoder(
            comprehensive_voxel_feature.permute(0, 1, 4, 2, 3))
        B = comprehensive_voxel_feature.shape[0]
        img_metas_occ = [{'pc_range': self.pc_range, 'occ_size': self.grid_size}
                         for _ in range(B)]
        occ_preds = self.prototype_query_decoder.simple_test(
            comprehensive_voxel_feature, img_metas_occ, mask_feat, prototoype_occ_pred)

        if self.proto_map_head is None:
            return occ_preds

        # ── Prototype Map Head ─────────────────────────────────────────────
        map_feature = self._select_map_feature(bev_feature, map_bev_feature)
        _, final_masks = self.proto_map_head(map_feature)
        map_probs = self.proto_map_head.predict(final_masks).detach().cpu().numpy()
        
        # If you want to use the coarse predictions, you can uncomment the following lines:
        # coarse_pred, final_masks = self.proto_map_head(map_feature)
        # map_probs = self.proto_map_head.predict(coarse_pred).detach().cpu().numpy()


        gt_masks_bev = self._normalize_map_targets(kwargs.get('gt_masks_bev'))
        if gt_masks_bev is not None:
            self._check_map_feature_size(map_feature, gt_masks_bev)

        if gt_masks_bev is not None:
            gt_masks_bev = gt_masks_bev.detach().cpu().numpy()

        results = []
        for i, occ_pred in enumerate(occ_preds):
            sample = {
                'occ_preds':  occ_pred,
                'masks_bev':  map_probs[i].astype(np.float32),
            }
            if gt_masks_bev is not None:
                sample['gt_masks_bev'] = gt_masks_bev[i].astype(np.int64)
            results.append(sample)
        return results

    def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        return None
