# Copyright (c) Phigent Robotics. All rights reserved.
import numpy as np

from .bevdet import BEVDet
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
import torch
from mmdet3d.models import builder

 
@DETECTORS.register_module()
class ProtoOccCnnSegHead(BEVDet):
    def __init__(self,
                 pc_range = [-40.0, -40.0, -1, 40.0, 40.0, 5.4],
                 grid_size = [200, 200, 16],
                 depth_net=None,
                 dual_branch_encoder=None,
                 cnn3d_decoder=None,
                 prototype_query_decoder=None,
                 bev_seg_head=None,
                 voxel_aware_map_ingest=None,
                 map_loss_weight=1.0,
                 **kwargs):
        super(ProtoOccCnnSegHead, self).__init__(**kwargs)
        self.pts_bbox_head = None # useless
        self.pc_range = torch.tensor(pc_range)
        self.grid_size = torch.tensor(grid_size)

        self.depth_net = build_head(depth_net)

        self.dual_branch_encoder = builder.build_backbone(dual_branch_encoder)
        self.cnn3d_decoder = build_head(cnn3d_decoder)
        self.prototype_query_decoder = build_head(prototype_query_decoder)
        self.bev_seg_head = build_head(bev_seg_head) if bev_seg_head is not None else None
        self.voxel_aware_map_ingest = (
            build_head(voxel_aware_map_ingest)
            if voxel_aware_map_ingest is not None else None)
        self.map_loss_weight = map_loss_weight

    def _needs_pqd_query_info(self):
        return (
            self.voxel_aware_map_ingest is not None and
            getattr(self.voxel_aware_map_ingest,
                    'requires_pqd_query_info',
                    False))

    def set_cfv_proto_tsfg_gamma(self, value):
        if (self.voxel_aware_map_ingest is not None and
                hasattr(self.voxel_aware_map_ingest, 'set_gamma')):
            self.voxel_aware_map_ingest.set_gamma(value)
            return True
        return False

    def _apply_voxel_aware_map_ingest(self,
                                      map_feature,
                                      voxel_feature,
                                      query_info=None):
        """Optional VAMI fusion before BEVSegHead.

        When ``voxel_aware_map_ingest`` is configured, route the
        OCC-supervised voxel feature into the map BEV feature. Otherwise
        return ``map_feature`` unchanged.
        """
        if self.voxel_aware_map_ingest is None:
            return map_feature
        query_info = query_info or {}
        return self.voxel_aware_map_ingest(
            map_feature=map_feature,
            voxel_feature=voxel_feature,
            **query_info)

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
        voxel_feat, depth, pv_feat = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        return voxel_feat, depth, pv_feat

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
                'BEV segmentation head requires either bev_feature or '
                'map_bev_feature from Dual_Branch_Encoder.')
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

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      voxel_semantics=None,
                      mask_camera = None,
                      sa_gt_depth=None,
                      sa_gt_semantic=None,
                      gt_masks_bev=None,
                      non_vis_semantic_voxel=None,
                      **kwargs):
        # 2D to 3D view transformation
        voxel_feat, depth, pv_feat = self.extract_feat(img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        # Dual Branch Encoder (DBE)
        encoder_output = self.dual_branch_encoder(voxel_feat)
        comprehensive_voxel_feature, bev_feature, map_bev_feature = \
            self._split_encoder_output(encoder_output)

        # 3d CNN for generating Prototype
        prototoype_occ_pred, mask_feat = self.cnn3d_decoder(comprehensive_voxel_feature.permute(0,1,4,2,3))
            
        # Prototype Query Decoder (PQD)
        B = comprehensive_voxel_feature.shape[0]
        img_metas = [{"pc_range": self.pc_range, "occ_size":self.grid_size} for i in range(B)]
        return_query_info = (
            self.bev_seg_head is not None and self._needs_pqd_query_info())
        pqd_outputs = self.prototype_query_decoder.forward_train(
            comprehensive_voxel_feature,
            img_metas,
            voxel_semantics,
            mask_camera,
            mask_feat,
            prototoype_occ_pred,
            return_query_info=return_query_info)
        if return_query_info:
            losses, query_info = pqd_outputs
        else:
            losses = pqd_outputs
            query_info = None
        
        # aggregate loss values
        loss_prototype = self.cnn3d_decoder.loss(prototoype_occ_pred, voxel_semantics, mask_camera)
        losses.update(loss_prototype)
        loss_depth = self.depth_net.get_PV_loss(pv_feat, depth, sa_gt_depth, sa_gt_semantic)
        losses.update(loss_depth)

        if self.bev_seg_head is not None:
            gt_masks_bev = self._normalize_map_targets(gt_masks_bev)
            if gt_masks_bev is None:
                raise ValueError('Expected `gt_masks_bev` when training ProtoOccCnnSegHead with a BEV segmentation head.')
            map_feature = self._select_map_feature(bev_feature, map_bev_feature)
            map_feature = self._apply_voxel_aware_map_ingest(
                map_feature, comprehensive_voxel_feature, query_info)
            self._check_map_feature_size(map_feature, gt_masks_bev)
            bev_seg_logits = self.bev_seg_head(map_feature)
            map_losses = self.bev_seg_head.loss(bev_seg_logits, gt_masks_bev)
            losses.update(self._scale_map_losses(map_losses))

        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # 2D to 3D view transformation
        voxel_feat, depth, pv_feat = self.extract_feat(img_inputs=img, img_metas=img_metas, **kwargs)

        # Dual Branch Encoder (DBE)
        encoder_output = self.dual_branch_encoder(voxel_feat)
        comprehensive_voxel_feature, bev_feature, map_bev_feature = \
            self._split_encoder_output(encoder_output)

        # Prototype Query Generator
        prototoype_occ_pred, mask_feat = self.cnn3d_decoder(comprehensive_voxel_feature.permute(0,1,4,2,3))

        # Prototype Query Decoder (PQD)
        B = comprehensive_voxel_feature.shape[0]
        img_metas_occ = [{"pc_range": self.pc_range, "occ_size":self.grid_size} for i in range(B)]
        return_query_info = (
            self.bev_seg_head is not None and self._needs_pqd_query_info())
        pqd_outputs = self.prototype_query_decoder.simple_test(
            comprehensive_voxel_feature,
            img_metas_occ,
            mask_feat,
            prototoype_occ_pred,
            return_query_info=return_query_info)
        if return_query_info:
            occ_preds, query_info = pqd_outputs
        else:
            occ_preds = pqd_outputs
            query_info = None

        if self.bev_seg_head is None:
            return occ_preds

        map_feature = self._select_map_feature(bev_feature, map_bev_feature)
        map_feature = self._apply_voxel_aware_map_ingest(
            map_feature, comprehensive_voxel_feature, query_info)
        bev_seg_probs = self.bev_seg_head.predict(self.bev_seg_head(map_feature)).detach().cpu().numpy()
        gt_masks_bev = self._normalize_map_targets(kwargs.get('gt_masks_bev'))
        if gt_masks_bev is not None:
            self._check_map_feature_size(map_feature, gt_masks_bev)
        if gt_masks_bev is not None:
            gt_masks_bev = gt_masks_bev.detach().cpu().numpy()

        results = []
        for batch_index, occ_pred in enumerate(occ_preds):
            sample_result = {
                'occ_preds': occ_pred,
                'masks_bev': bev_seg_probs[batch_index].astype(np.float32),
            }
            if gt_masks_bev is not None:
                sample_result['gt_masks_bev'] = gt_masks_bev[batch_index].astype(np.int64)
            results.append(sample_result)
        return results

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        return None
