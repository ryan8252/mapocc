import numpy as np
import torch

from mmdet3d.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.builder import build_head

from .bevdet import BEVDet


@DETECTORS.register_module()
class MAESTRO2Task(BEVDet):
    """Camera-only MAESTRO 2-task reproduction for occupancy and BEV maps.

    This detector intentionally removes ProtoOcc's Dual_Branch_Encoder, CNN3D
    decoder, and PQD. The shared feature is the voxel tensor produced directly
    by LSSViewTransformer_depthGT; MAESTRO CPG/TSFG/SPA then creates task
    features for the map head and OccFormer-style occupancy head.
    """

    def __init__(self,
                 pc_range=[-40.0, -40.0, -1, 40.0, 40.0, 5.4],
                 grid_size=[200, 200, 16],
                 occ_pc_range=None,
                 occ_grid_size=None,
                 depth_net=None,
                 maestro_cpg=None,
                 maestro_map_tsfg=None,
                 maestro_occ_tsfg=None,
                 maestro_spa=None,
                 bev_seg_head=None,
                 occ_head=None,
                 map_loss_weight=1.0,
                 free_label=17,
                 **kwargs):
        super(MAESTRO2Task, self).__init__(**kwargs)
        self.pts_bbox_head = None
        self.pc_range = pc_range
        self.grid_size = grid_size
        self.occ_pc_range = occ_pc_range or pc_range
        self.occ_grid_size = occ_grid_size or grid_size
        self._validate_occ_crop_config()
        self.depth_net = build_head(depth_net)
        self.maestro_cpg = builder.build_backbone(maestro_cpg)
        self.maestro_map_tsfg = builder.build_backbone(maestro_map_tsfg)
        self.maestro_occ_tsfg = builder.build_backbone(maestro_occ_tsfg)
        self.maestro_spa = (
            builder.build_backbone(maestro_spa)
            if maestro_spa is not None else None)
        self.bev_seg_head = build_head(bev_seg_head)
        self.occ_head = build_head(occ_head)
        self.map_loss_weight = map_loss_weight
        self.free_label = free_label

    def image_encoder(self, img, stereo=False):
        imgs = img
        batch_size, num_cams, channels, height, width = imgs.shape
        imgs = imgs.view(batch_size * num_cams, channels, height, width)
        feats = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = feats[0]
            feats = feats[1:]
        if self.with_img_neck:
            feats = self.img_neck(feats)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
        _, out_channels, out_height, out_width = feats.shape
        feats = feats.view(
            batch_size, num_cams, out_channels, out_height, out_width)
        return feats, stereo_feat

    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        img_inputs = self.prepare_inputs(img_inputs)
        img_feats, _ = self.image_encoder(img_inputs[0])
        cam_params = img_inputs[1:7]
        mlp_input = self.depth_net.get_mlp_input(*cam_params)
        pv_feat, depth = self.depth_net(img_feats, mlp_input)
        voxel_feat, depth = self.img_view_transformer(
            depth, pv_feat, [img_feats] + img_inputs[1:7])
        return voxel_feat, depth, pv_feat

    def extract_feat(self, img_inputs, img_metas, **kwargs):
        return self.extract_img_feat(img_inputs, img_metas, **kwargs)

    def _to_maestro_layout(self, voxel_feat):
        if voxel_feat.dim() != 5:
            raise ValueError(
                f'Expected 5D LSS voxel feature, got {voxel_feat.shape}.')

        expected_z = int(self.grid_size[2])
        if voxel_feat.shape[2] == expected_z:
            return voxel_feat.permute(0, 1, 4, 3, 2).contiguous()
        if voxel_feat.shape[-1] == expected_z:
            return voxel_feat.contiguous()
        raise ValueError(
            'Cannot infer LSS voxel layout: expected either '
            f'dim2 or dim4 to equal grid Z={expected_z}, got '
            f'{tuple(voxel_feat.shape)}.')

    def _xy_voxel_size(self, pc_range, grid_size):
        return (
            (float(pc_range[3]) - float(pc_range[0])) / int(grid_size[0]),
            (float(pc_range[4]) - float(pc_range[1])) / int(grid_size[1]),
        )

    def _xy_center(self, pc_range):
        return (
            (float(pc_range[0]) + float(pc_range[3])) * 0.5,
            (float(pc_range[1]) + float(pc_range[4])) * 0.5,
        )

    def _validate_occ_crop_config(self):
        shared_x, shared_y = int(self.grid_size[0]), int(self.grid_size[1])
        occ_x, occ_y = int(self.occ_grid_size[0]), int(self.occ_grid_size[1])
        if shared_x < occ_x or shared_y < occ_y:
            raise ValueError(
                f'occ_grid_size {self.occ_grid_size} cannot exceed shared '
                f'grid_size {self.grid_size}.')
        if (shared_x - occ_x) % 2 != 0 or (shared_y - occ_y) % 2 != 0:
            raise ValueError(
                'MAESTRO2Task only supports centered OCC crops with even '
                f'spatial margins, got shared={self.grid_size}, '
                f'occ={self.occ_grid_size}.')

        shared_voxel = self._xy_voxel_size(self.pc_range, self.grid_size)
        occ_voxel = self._xy_voxel_size(self.occ_pc_range, self.occ_grid_size)
        if any(abs(a - b) > 1e-6 for a, b in zip(shared_voxel, occ_voxel)):
            raise ValueError(
                'Shared and OCC ranges must use the same XY voxel size for a '
                f'clean center crop, got shared={shared_voxel}, '
                f'occ={occ_voxel}.')
        if any(abs(a - b) > 1e-6 for a, b in zip(
                self._xy_center(self.pc_range),
                self._xy_center(self.occ_pc_range))):
            raise ValueError(
                'Shared and OCC ranges must share the same XY center for '
                f'center crop alignment, got shared={self.pc_range}, '
                f'occ={self.occ_pc_range}.')

    def _get_occ_crop_slices(self):
        shared_x, shared_y = int(self.grid_size[0]), int(self.grid_size[1])
        occ_x, occ_y = int(self.occ_grid_size[0]), int(self.occ_grid_size[1])
        if shared_x == occ_x and shared_y == occ_y:
            return None

        x0 = (shared_x - occ_x) // 2
        y0 = (shared_y - occ_y) // 2
        return slice(x0, x0 + occ_x), slice(y0, y0 + occ_y)

    def _crop_voxel_xy_to_occ(self, voxel_tensor, crop_slices):
        if crop_slices is None or voxel_tensor is None:
            return voxel_tensor
        x_slice, y_slice = crop_slices
        return voxel_tensor[:, :, x_slice, y_slice, :]

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

    def _scale_map_losses(self, map_losses):
        if self.map_loss_weight == 1.0:
            return map_losses
        return {
            name: loss * self.map_loss_weight
            for name, loss in map_losses.items()
        }

    def _make_map_relevance_target(self, gt_masks_bev):
        if gt_masks_bev is None:
            return None
        return gt_masks_bev.max(dim=1).values.float()

    def _make_occ_relevance_target(self, voxel_semantics, mask_camera=None):
        if voxel_semantics is None:
            return None
        target = (voxel_semantics.long() != self.free_label)
        target = target & (voxel_semantics.long() != 255)
        if mask_camera is not None:
            target = target & mask_camera.bool()
        return target.float()

    def _make_occ_img_metas(self, batch_size):
        return [
            {
                'pc_range': self.occ_pc_range,
                'occ_size': self.occ_grid_size,
            } for _ in range(batch_size)
        ]

    def _run_maestro_branches(self,
                              Shared_Voxel_Feature,
                              voxel_semantics=None,
                              mask_camera=None,
                              gt_masks_bev=None):
        # Class-wise Prototype Generator (CPG): run once on F_s.
        cpg_output = self.maestro_cpg(Shared_Voxel_Feature)
        occ_crop_slices = self._get_occ_crop_slices()

        # Map TSFG receives the background prototype group P_bg.
        map_target = self._make_map_relevance_target(gt_masks_bev)
        map_feature, map_aux_losses = self.maestro_map_tsfg(
            Shared_Voxel_Feature,
            cpg_output['background'],
            target_mask=map_target,
        )
        map_logits = self.bev_seg_head(map_feature)

        if self.maestro_spa is not None:
            background_for_occ = self.maestro_spa(
                cpg_output['background'], map_feature, map_logits)
        else:
            background_for_occ = cpg_output['background']

        # Occupancy TSFG receives G_occ = [P_fg, P_bg].
        occ_prototypes = torch.cat(
            [cpg_output['foreground'], background_for_occ], dim=1)
        occ_target = self._make_occ_relevance_target(
            voxel_semantics, mask_camera)
        occ_feature, occ_aux_losses = self.maestro_occ_tsfg(
            Shared_Voxel_Feature,
            occ_prototypes,
            target_mask=occ_target,
            target_crop_slices=occ_crop_slices,
        )
        occ_feature = self._crop_voxel_xy_to_occ(
            occ_feature, occ_crop_slices)
        cpg_logits = self._crop_voxel_xy_to_occ(
            cpg_output['logits'], occ_crop_slices)

        aux_losses = {}
        aux_losses.update(map_aux_losses)
        aux_losses.update(occ_aux_losses)
        # CPG loss reuses S_v from cpg_output; it does not run CPG again.
        aux_losses.update(
            self.maestro_cpg.loss(
                cpg_logits,
                voxel_semantics,
                valid_mask=mask_camera,
            ))
        return occ_feature, map_feature, map_logits, occ_prototypes, aux_losses

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
        # Shared Backbone
        voxel_feat, depth, pv_feat = self.extract_feat(img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        Shared_Voxel_Feature = self._to_maestro_layout(voxel_feat)

        gt_masks_bev = self._normalize_map_targets(gt_masks_bev)
        if gt_masks_bev is None:
            raise ValueError('Expected `gt_masks_bev` when training MAESTRO2Task.')

        (occ_feature, map_feature, map_logits, occ_prototypes,
         aux_losses) = self._run_maestro_branches(
            Shared_Voxel_Feature,
            voxel_semantics=voxel_semantics,
            mask_camera=mask_camera,
            gt_masks_bev=gt_masks_bev,
        )
        self._check_map_feature_size(map_feature, gt_masks_bev)

        batch_size = occ_feature.shape[0]
        img_metas_occ = self._make_occ_img_metas(batch_size)
        losses = self.occ_head.forward_train(
            occ_feature,
            img_metas_occ,
            voxel_semantics,
            mask_camera=mask_camera,
            scene_prototypes=occ_prototypes,
        )
        losses.update(
            self.depth_net.get_PV_loss(
                pv_feat, depth, sa_gt_depth, sa_gt_semantic))
        losses.update(
            self._scale_map_losses(
                self.bev_seg_head.loss(map_logits, gt_masks_bev)))
        losses.update(aux_losses)
        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        voxel_feat, depth, pv_feat = self.extract_feat(
            img_inputs=img, img_metas=img_metas, **kwargs)
        Shared_Voxel_Feature = self._to_maestro_layout(voxel_feat)
        gt_masks_bev = self._normalize_map_targets(kwargs.get('gt_masks_bev'))

        (occ_feature, map_feature, map_logits, occ_prototypes,
         _) = self._run_maestro_branches(
            Shared_Voxel_Feature,
            gt_masks_bev=gt_masks_bev,
        )

        batch_size = occ_feature.shape[0]
        img_metas_occ = self._make_occ_img_metas(batch_size)
        occ_preds = self.occ_head.simple_test(
            occ_feature, img_metas_occ, scene_prototypes=occ_prototypes)
        map_probs = self.bev_seg_head.predict(map_logits).detach().cpu().numpy()

        if gt_masks_bev is not None:
            self._check_map_feature_size(map_feature, gt_masks_bev)
            gt_masks_bev = gt_masks_bev.detach().cpu().numpy()

        results = []
        for batch_index, occ_pred in enumerate(occ_preds):
            sample = {
                'occ_preds': occ_pred,
                'masks_bev': map_probs[batch_index].astype(np.float32),
            }
            if gt_masks_bev is not None:
                sample['gt_masks_bev'] = gt_masks_bev[batch_index].astype(np.int64)
            results.append(sample)
        return results

    def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        return None
