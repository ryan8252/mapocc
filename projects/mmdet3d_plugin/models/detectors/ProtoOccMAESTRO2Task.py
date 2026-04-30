import numpy as np
import torch

from mmdet3d.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.builder import build_head

from .ProtoOcc import ProtoOcc


@DETECTORS.register_module()
class ProtoOccMAESTRO2Task(ProtoOcc):
    """ProtoOcc wrapper with MAESTRO-style map and occupancy branches.

    The wrapper keeps ProtoOcc's image encoder, view transformer, DBE, CNN
    prototype decoder, and PQD. MAESTRO modules are inserted after DBE to
    create task-specific features for BEV map segmentation and occupancy.
    """

    def __init__(self,
                 pc_range=[-40.0, -40.0, -1, 40.0, 40.0, 5.4],
                 grid_size=[200, 200, 16],
                 depth_net=None,
                 dual_branch_encoder=None,
                 cnn3d_decoder=None,
                 prototype_query_decoder=None,
                 maestro_cpg=None,
                 maestro_map_tsfg=None,
                 maestro_occ_tsfg=None,
                 maestro_spa=None,
                 bev_seg_head=None,
                 map_loss_weight=1.0,
                 free_label=17,
                 **kwargs):
        super(ProtoOccMAESTRO2Task, self).__init__(
            pc_range=pc_range,
            grid_size=grid_size,
            depth_net=depth_net,
            dual_branch_encoder=dual_branch_encoder,
            cnn3d_decoder=cnn3d_decoder,
            prototype_query_decoder=prototype_query_decoder,
            **kwargs)

        self.maestro_cpg = builder.build_backbone(maestro_cpg)
        self.maestro_map_tsfg = builder.build_backbone(maestro_map_tsfg)
        self.maestro_occ_tsfg = builder.build_backbone(maestro_occ_tsfg)
        self.maestro_spa = (
            builder.build_backbone(maestro_spa)
            if maestro_spa is not None else None)
        self.bev_seg_head = build_head(bev_seg_head)
        self.map_loss_weight = map_loss_weight
        self.free_label = free_label

    def _split_encoder_output(self, encoder_output):
        if isinstance(encoder_output, tuple):
            return encoder_output[0]
        return encoder_output

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
        return {name: loss * self.map_loss_weight
                for name, loss in map_losses.items()}

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

    def _run_maestro_branches(self,
                              shared_feature,
                              voxel_semantics=None,
                              mask_camera=None,
                              gt_masks_bev=None):
        cpg_output = self.maestro_cpg(shared_feature)

        map_target = self._make_map_relevance_target(gt_masks_bev)
        map_feature, map_aux_losses = self.maestro_map_tsfg(
            shared_feature,
            cpg_output['background'],
            target_mask=map_target,
        )
        map_logits = self.bev_seg_head(map_feature)

        if self.maestro_spa is not None:
            background_for_occ = self.maestro_spa(
                cpg_output['background'], map_feature, map_logits)
        else:
            background_for_occ = cpg_output['background']

        occ_prototypes = torch.cat(
            [cpg_output['foreground'], background_for_occ], dim=1)
        occ_target = self._make_occ_relevance_target(voxel_semantics, mask_camera)
        occ_feature, occ_aux_losses = self.maestro_occ_tsfg(
            shared_feature,
            occ_prototypes,
            target_mask=occ_target,
        )

        aux_losses = {}
        aux_losses.update(map_aux_losses)
        aux_losses.update(occ_aux_losses)
        aux_losses.update(
            self.maestro_cpg.loss(
                cpg_output['logits'],
                voxel_semantics,
                valid_mask=mask_camera,
            ))

        return occ_feature, map_feature, map_logits, aux_losses

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
        voxel_feat, depth, pv_feat = self.extract_feat(
            img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        shared_feature = self._split_encoder_output(
            self.dual_branch_encoder(voxel_feat))

        gt_masks_bev = self._normalize_map_targets(gt_masks_bev)
        if gt_masks_bev is None:
            raise ValueError(
                'Expected `gt_masks_bev` when training ProtoOccMAESTRO2Task.')

        occ_feature, map_feature, map_logits, aux_losses = self._run_maestro_branches(
            shared_feature,
            voxel_semantics=voxel_semantics,
            mask_camera=mask_camera,
            gt_masks_bev=gt_masks_bev,
        )

        self._check_map_feature_size(map_feature, gt_masks_bev)

        prototype_occ_pred, mask_feat = self.cnn3d_decoder(
            occ_feature.permute(0, 1, 4, 2, 3))

        batch_size = occ_feature.shape[0]
        img_metas_occ = [
            {'pc_range': self.pc_range, 'occ_size': self.grid_size}
            for _ in range(batch_size)
        ]
        losses = self.prototype_query_decoder.forward_train(
            occ_feature,
            img_metas_occ,
            voxel_semantics,
            mask_camera,
            mask_feat,
            prototype_occ_pred,
        )
        losses.update(
            self.cnn3d_decoder.loss(
                prototype_occ_pred, voxel_semantics, mask_camera))
        losses.update(
            self.depth_net.get_PV_loss(
                pv_feat, depth, sa_gt_depth, sa_gt_semantic))

        map_losses = self.bev_seg_head.loss(map_logits, gt_masks_bev)
        losses.update(self._scale_map_losses(map_losses))
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

        shared_feature = self._split_encoder_output(
            self.dual_branch_encoder(voxel_feat))

        gt_masks_bev = self._normalize_map_targets(kwargs.get('gt_masks_bev'))
        occ_feature, map_feature, map_logits, _ = self._run_maestro_branches(
            shared_feature,
            gt_masks_bev=gt_masks_bev,
        )

        prototype_occ_pred, mask_feat = self.cnn3d_decoder(
            occ_feature.permute(0, 1, 4, 2, 3))

        batch_size = occ_feature.shape[0]
        img_metas_occ = [
            {'pc_range': self.pc_range, 'occ_size': self.grid_size}
            for _ in range(batch_size)
        ]
        occ_preds = self.prototype_query_decoder.simple_test(
            occ_feature, img_metas_occ, mask_feat, prototype_occ_pred)

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

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        return None
