# Copyright (c) Phigent Robotics. All rights reserved.
import numpy as np

from .bevdet import BEVDet
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
import torch
from torch import nn
import torch.nn.functional as F
from mmdet3d.models import builder


class TaskChannelScaling(nn.Module):
    """Lightweight task-aware channel scaling for OCC and map features.

    The last linear layer is zero-initialized, so both branches start as exact
    identity transforms: feature * (1 + max_residual * tanh(0)).
    """

    def __init__(self,
                 occ_channels,
                 map_channels,
                 reduction=4,
                 min_hidden_channels=16,
                 max_residual=0.5,
                 detach_context=False):
        super().__init__()
        self.occ_channels = int(occ_channels)
        self.map_channels = int(map_channels)
        self.max_residual = float(max_residual)
        self.detach_context = bool(detach_context)
        self.occ_gate = self._make_gate(
            self.occ_channels, reduction, min_hidden_channels)
        self.map_gate = self._make_gate(
            self.map_channels, reduction, min_hidden_channels)

    @staticmethod
    def _make_gate(channels, reduction, min_hidden_channels):
        hidden_channels = max(
            int(channels) // max(int(reduction), 1),
            int(min_hidden_channels))
        gate = nn.Sequential(
            nn.Linear(int(channels), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, int(channels)))
        nn.init.zeros_(gate[-1].weight)
        nn.init.zeros_(gate[-1].bias)
        return gate

    def _context(self, feature, dims):
        source = feature.detach() if self.detach_context else feature
        return source.mean(dim=dims)

    def scale_occ(self, occ_feature):
        if occ_feature.shape[1] != self.occ_channels:
            raise ValueError(
                'TaskChannelScaling occ channel mismatch: expected '
                f'{self.occ_channels}, got {occ_feature.shape[1]}.')
        context = self._context(occ_feature, dims=(2, 3, 4))
        scale = 1.0 + self.max_residual * torch.tanh(
            self.occ_gate(context))
        return occ_feature * scale.view(scale.shape[0], -1, 1, 1, 1)

    def scale_map(self, map_feature):
        if map_feature.shape[1] != self.map_channels:
            raise ValueError(
                'TaskChannelScaling map channel mismatch: expected '
                f'{self.map_channels}, got {map_feature.shape[1]}.')
        context = self._context(map_feature, dims=(2, 3))
        scale = 1.0 + self.max_residual * torch.tanh(
            self.map_gate(context))
        return map_feature * scale.view(scale.shape[0], -1, 1, 1)


class LossUncertaintyWeighting(nn.Module):
    """Homoscedastic uncertainty weighting over named loss groups."""

    DEFAULT_GROUPS = (
        dict(name='map', prefixes=('loss_map_', )),
        dict(name='depth_img',
             keys=('loss_depth', 'loss_segmentation')),
        dict(name='occ_proto',
             keys=('loss_CE_prototype',
                   'lovasz_softmax_loss_prototype')),
        dict(name='pqd',
             keys=('loss_cls', 'loss_mask', 'loss_dice',
                   'loss_cls_RPL', 'loss_mask_RPL', 'loss_dice_RPL')),
    )

    def __init__(self,
                 groups=None,
                 init_log_vars=None,
                 min_log_var=-2.0,
                 max_log_var=2.0):
        super().__init__()
        groups = groups if groups is not None else self.DEFAULT_GROUPS
        init_log_vars = init_log_vars or {}
        self.groups = []
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)
        self.log_vars = nn.ParameterDict()
        for group in groups:
            name = group['name']
            keys = tuple(group.get('keys', ()))
            prefixes = tuple(group.get('prefixes', ()))
            self.groups.append(dict(name=name, keys=keys, prefixes=prefixes))
            init_value = float(init_log_vars.get(name, 0.0))
            self.log_vars[name] = nn.Parameter(torch.tensor([init_value]))

    @staticmethod
    def _scale_value(value, scale):
        if torch.is_tensor(value):
            return value * scale
        if isinstance(value, (list, tuple)):
            return [item * scale for item in value]
        return value

    @staticmethod
    def _mean_value(value):
        if torch.is_tensor(value):
            return value.mean()
        if isinstance(value, (list, tuple)):
            total = None
            for item in value:
                item_mean = item.mean()
                total = item_mean if total is None else total + item_mean
            return total
        return None

    @staticmethod
    def _matches(loss_name, group):
        if loss_name in group['keys']:
            return True
        return any(loss_name.startswith(prefix)
                   for prefix in group['prefixes'])

    def forward(self, losses):
        weighted = dict(losses)
        assigned = set()
        for group in self.groups:
            name = group['name']
            group_keys = [
                key for key in losses
                if key not in assigned and self._matches(key, group)
            ]
            if not group_keys:
                continue

            log_var = torch.clamp(
                self.log_vars[name], self.min_log_var, self.max_log_var)
            precision = torch.exp(-log_var)
            raw_total = None
            for key in group_keys:
                value = losses[key]
                weighted[key] = self._scale_value(value, precision)
                value_mean = self._mean_value(value)
                if value_mean is not None:
                    raw_total = (
                        value_mean if raw_total is None
                        else raw_total + value_mean)
                assigned.add(key)

            weighted[f'loss_uncertainty_{name}_reg'] = log_var.squeeze()
            weighted[f'uncertainty_{name}_precision'] = (
                precision.detach().squeeze())
            weighted[f'uncertainty_{name}_log_var'] = (
                log_var.detach().squeeze())
            if raw_total is not None:
                weighted[f'uncertainty_{name}_raw'] = raw_total.detach()
        return weighted


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
                 map_img_view_transformer=None,
                 map_depth_net=None,
                 map_lss2d_encoder=None,
                 map_lss2d_fusion=None,
                 use_lss2d_as_bev_branch=False,
                 voxel_aware_map_ingest=None,
                 task_channel_scaling=None,
                 shared_feature_range=None,
                 occ_feature_range=None,
                 map_feature_range=None,
                 map_feature_size=None,
                 map_loss_weight=1.0,
                 use_occ2map_prior=False,
                 occ2map_prior_detach=True,
                 occ2map_prior_feature_range=None,
                 occ2map_prior_context_z=3,
                 occ2map_prior_output_z_indices=(0, 1),
                 occ2map_prior_class_ids=(11, 13),
                 loss_uncertainty_weighting=None,
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
        self.map_img_view_transformer = (
            builder.build_neck(map_img_view_transformer)
            if map_img_view_transformer is not None else None)
        # Optional separate, *unsupervised* depth for the map LSS branch so it is
        # decoupled from the OCC (depth-supervised, sharp) depth distribution.
        self.map_depth_net = (
            build_head(map_depth_net) if map_depth_net is not None else None)
        if self.map_depth_net is not None and self.map_img_view_transformer is None:
            raise ValueError(
                'map_depth_net requires map_img_view_transformer.')
        self.map_lss2d_encoder = (
            builder.build_backbone(map_lss2d_encoder)
            if map_lss2d_encoder is not None else None)
        self.map_lss2d_fusion = (
            builder.build_backbone(map_lss2d_fusion)
            if map_lss2d_fusion is not None else None)
        self.use_lss2d_as_bev_branch = bool(use_lss2d_as_bev_branch)
        needs_map_lss2d = (
            self.map_lss2d_encoder is not None
            or self.map_lss2d_fusion is not None
            or self.use_lss2d_as_bev_branch)
        if needs_map_lss2d and self.map_img_view_transformer is None:
            raise ValueError(
                'map_lss2d_encoder/map_lss2d_fusion/'
                'use_lss2d_as_bev_branch requires map_img_view_transformer.')
        self.voxel_aware_map_ingest = (
            build_head(voxel_aware_map_ingest)
            if voxel_aware_map_ingest is not None else None)
        self.task_channel_scaling = (
            TaskChannelScaling(**task_channel_scaling)
            if task_channel_scaling is not None else None)
        self.loss_uncertainty_weighting = (
            LossUncertaintyWeighting(**loss_uncertainty_weighting)
            if loss_uncertainty_weighting is not None else None)
        self.shared_feature_range = torch.tensor(
            shared_feature_range if shared_feature_range is not None else pc_range,
            dtype=torch.float32)
        self.occ_feature_range = torch.tensor(
            occ_feature_range if occ_feature_range is not None else pc_range,
            dtype=torch.float32)
        self.map_feature_range = (
            None if map_feature_range is None
            else torch.tensor(map_feature_range, dtype=torch.float32))
        self.map_feature_size = (
            None if map_feature_size is None
            else tuple(int(v) for v in map_feature_size))
        self.map_loss_weight = map_loss_weight
        self.use_occ2map_prior = bool(use_occ2map_prior)
        self.occ2map_prior_detach = bool(occ2map_prior_detach)
        if occ2map_prior_feature_range is None:
            if map_feature_range is not None:
                occ2map_prior_feature_range = [
                    map_feature_range[0],
                    map_feature_range[1],
                    float(self.shared_feature_range[2]),
                    map_feature_range[2],
                    map_feature_range[3],
                    float(self.shared_feature_range[5]),
                ]
            else:
                occ2map_prior_feature_range = self.shared_feature_range.tolist()
        self.occ2map_prior_feature_range = torch.tensor(
            occ2map_prior_feature_range, dtype=torch.float32)
        self.occ2map_prior_context_z = int(occ2map_prior_context_z)
        self.occ2map_prior_output_z_indices = tuple(
            int(index) for index in occ2map_prior_output_z_indices)
        self.occ2map_prior_class_ids = tuple(
            int(index) for index in occ2map_prior_class_ids)
        self._validate_occ2map_prior_cfg()

    def _validate_occ2map_prior_cfg(self):
        if not self.use_occ2map_prior:
            return
        if self.bev_seg_head is None:
            raise ValueError(
                'use_occ2map_prior=True requires a configured bev_seg_head.')
        if self.map_feature_range is None or self.map_feature_size is None:
            raise ValueError(
                'use_occ2map_prior=True requires map_feature_range and '
                'map_feature_size so the prior can be aligned to map BEV.')
        if self.occ2map_prior_context_z <= 0:
            raise ValueError('occ2map_prior_context_z must be positive.')
        if not self.occ2map_prior_output_z_indices:
            raise ValueError(
                'occ2map_prior_output_z_indices must contain at least one '
                'z index.')
        if any(index < 0 for index in self.occ2map_prior_output_z_indices):
            raise ValueError(
                'occ2map_prior_output_z_indices must be non-negative.')
        if len(self.occ2map_prior_class_ids) != 2:
            raise ValueError(
                'occ2map_prior_class_ids must be '
                '[driveable_surface_id, sidewalk_id].')
        if any(index < 0 for index in self.occ2map_prior_class_ids):
            raise ValueError('occ2map_prior_class_ids must be non-negative.')
        occ_num_classes = getattr(self.cnn3d_decoder, 'num_classes', None)
        if (occ_num_classes is not None and
                max(self.occ2map_prior_class_ids) >= occ_num_classes):
            raise ValueError(
                'occ2map_prior_class_ids contains an index outside '
                f'cnn3d_decoder.num_classes={occ_num_classes}.')

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
                                      query_info=None,
                                      gt_masks_bev=None):
        """Optional VAMI fusion before BEVSegHead.

        When ``voxel_aware_map_ingest`` is configured, route the
        OCC-supervised voxel feature into the map BEV feature. Otherwise
        return ``map_feature`` unchanged.
        """
        if self.voxel_aware_map_ingest is None:
            return map_feature, {}
        query_info = query_info or {}
        module = self.voxel_aware_map_ingest
        returns_aux = getattr(module, 'returns_aux_losses', False)

        def _run(feat):
            if returns_aux:
                return module(
                    map_feature=feat,
                    voxel_feature=voxel_feature,
                    gt_masks_bev=gt_masks_bev,
                    **query_info)
            return module(
                map_feature=feat,
                voxel_feature=voxel_feature,
                **query_info), {}

        if isinstance(map_feature, dict):
            updated = dict(map_feature)
            updated_feature, aux_losses = _run(
                self._get_map_feature_tensor(map_feature))
            updated['bev_feature'] = updated_feature
            return updated, aux_losses
        return _run(map_feature)

    def _apply_task_channel_scaling_occ(self, occ_feature):
        if self.task_channel_scaling is None:
            return occ_feature
        return self.task_channel_scaling.scale_occ(occ_feature)

    def _apply_task_channel_scaling_map(self, map_feature):
        if self.task_channel_scaling is None:
            return map_feature
        if isinstance(map_feature, dict):
            scaled = dict(map_feature)
            scaled['bev_feature'] = self.task_channel_scaling.scale_map(
                self._get_map_feature_tensor(map_feature))
            return scaled
        return self.task_channel_scaling.scale_map(map_feature)

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
        
    def _extract_img_feat_impl(self, img_inputs, img_metas, **kwargs):
        img_inputs = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(img_inputs[0])
        cam_params = img_inputs[1:7]

        mlp_input = self.depth_net.get_mlp_input(*cam_params)
        pv_feat, depth = self.depth_net(x, mlp_input)
        voxel_feat, depth = self.img_view_transformer(depth, pv_feat, [x] + img_inputs[1:7])
        map_lss2d_feat = None
        if self.map_img_view_transformer is not None:
            # Map branch uses its own soft, unsupervised depth when configured;
            # otherwise it falls back to the shared (OCC, sharp) depth.
            map_depth = depth
            if self.map_depth_net is not None:
                map_depth = self.map_depth_net(x)
            map_lss2d_feat, _ = self.map_img_view_transformer(
                map_depth, pv_feat, [x] + img_inputs[1:7])

        return voxel_feat, depth, pv_feat, map_lss2d_feat

    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        voxel_feat, depth, pv_feat, _ = self._extract_img_feat_impl(
            img_inputs, img_metas, **kwargs)
        return voxel_feat, depth, pv_feat

    def extract_feat(self, img_inputs, img_metas, **kwargs):
        voxel_feat, depth, pv_feat = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        return voxel_feat, depth, pv_feat

    def extract_feat_with_map_lss(self, img_inputs, img_metas, **kwargs):
        return self._extract_img_feat_impl(img_inputs, img_metas, **kwargs)

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

    def _select_map_feature(self,
                            bev_feature,
                            map_bev_feature,
                            lss2d_map_feature=None):
        if lss2d_map_feature is not None:
            map_feature = lss2d_map_feature
        elif map_bev_feature is not None:
            map_feature = map_bev_feature
        else:
            map_feature = bev_feature
        if map_feature is None:
            raise ValueError(
                'BEV segmentation head requires either bev_feature or '
                'map_bev_feature from Dual_Branch_Encoder, or a '
                'map_lss2d_encoder output.')
        return map_feature

    def _get_map_feature_tensor(self, map_feature):
        if isinstance(map_feature, dict):
            map_tensor = map_feature.get('bev_feature', None)
            if map_tensor is None:
                raise ValueError('map_feature dict must contain `bev_feature`.')
            return map_tensor
        return map_feature

    def _encode_lss2d_map_feature(self, map_lss2d_feat):
        if map_lss2d_feat is None:
            return None
        if self.map_lss2d_encoder is not None:
            return self.map_lss2d_encoder(map_lss2d_feat)
        if self.map_lss2d_fusion is not None:
            return map_lss2d_feat
        return None

    def _fuse_lss2d_map_feature(self, map_feature, lss2d_map_feature):
        if self.map_lss2d_fusion is None or lss2d_map_feature is None:
            return map_feature

        lss2d_tensor = self._get_map_feature_tensor(lss2d_map_feature)
        if isinstance(map_feature, dict):
            fused = dict(map_feature)
            fused['bev_feature'] = self.map_lss2d_fusion(
                self._get_map_feature_tensor(map_feature), lss2d_tensor)
            return fused
        return self.map_lss2d_fusion(map_feature, lss2d_tensor)

    def _prepare_map_feature(self,
                             bev_feature,
                             map_bev_feature,
                             lss2d_map_feature=None):
        if self.map_lss2d_fusion is not None and lss2d_map_feature is not None:
            map_feature = self._select_map_feature(
                bev_feature, map_bev_feature, None)
            map_feature = self._align_map_feature(map_feature)
            lss2d_map_feature = self._align_map_feature(lss2d_map_feature)
            return self._fuse_lss2d_map_feature(
                map_feature, lss2d_map_feature)

        map_feature = self._select_map_feature(
            bev_feature, map_bev_feature, lss2d_map_feature)
        return self._align_map_feature(map_feature)

    def _range_to_xyz(self, value):
        if value.numel() == 6:
            return value
        if value.numel() == 4:
            return torch.tensor(
                [value[0], value[1], 0.0, value[2], value[3], 1.0],
                dtype=value.dtype,
                device=value.device)
        raise ValueError(
            'Expected a range with 4 BEV values or 6 XYZ values, '
            f'but got {value.numel()} values.')

    def _feature_slices(self, feature_shape, source_range, target_range):
        source_range = self._range_to_xyz(source_range)
        target_range = self._range_to_xyz(target_range)
        slices = []
        for axis, size in enumerate(feature_shape):
            source_min = float(source_range[axis])
            source_max = float(source_range[axis + 3])
            target_min = float(target_range[axis])
            target_max = float(target_range[axis + 3])
            if abs(source_min - target_min) < 1e-6 and abs(source_max - target_max) < 1e-6:
                slices.append(slice(None))
                continue
            step = (source_max - source_min) / float(size)
            start = int(round((target_min - source_min) / step))
            end = int(round((target_max - source_min) / step))
            if start < 0 or end > size or start >= end:
                raise ValueError(
                    'Target feature range must be inside source range. '
                    f'axis={axis}, source=({source_min}, {source_max}), '
                    f'target=({target_min}, {target_max}), size={size}.')
            slices.append(slice(start, end))
        return slices

    def _crop_occ_feature(self, voxel_feature):
        source_range = self.shared_feature_range.to(voxel_feature.device)
        target_range = self.occ_feature_range.to(voxel_feature.device)
        slices = self._feature_slices(
            voxel_feature.shape[2:5], source_range, target_range)
        return voxel_feature[
            :,
            :,
            slices[0],
            slices[1],
            slices[2],
        ]

    def _crop_occ2map_prior_feature(self, voxel_feature):
        source_range = self.shared_feature_range.to(voxel_feature.device)
        target_range = self.occ2map_prior_feature_range.to(
            voxel_feature.device)
        slices = self._feature_slices(
            voxel_feature.shape[2:5], source_range, target_range)
        z_slice = slices[2]
        z_start = 0 if z_slice.start is None else z_slice.start
        z_stop = voxel_feature.shape[4] if z_slice.stop is None else z_slice.stop
        z_stop = min(z_stop, z_start + self.occ2map_prior_context_z)
        if z_stop <= z_start:
            raise ValueError(
                'occ2map prior z crop is empty. Check '
                'occ2map_prior_feature_range and occ2map_prior_context_z.')
        return voxel_feature[
            :,
            :,
            slices[0],
            slices[1],
            slice(z_start, z_stop),
        ]

    def _make_occ2map_prior(self, voxel_feature):
        if not self.use_occ2map_prior:
            return None

        lowz_feature = self._crop_occ2map_prior_feature(voxel_feature)
        if self.occ2map_prior_detach:
            with torch.no_grad():
                occ_logits, _ = self.cnn3d_decoder(
                    lowz_feature.permute(0, 1, 4, 2, 3))
        else:
            occ_logits, _ = self.cnn3d_decoder(
                lowz_feature.permute(0, 1, 4, 2, 3))

        z_dim = occ_logits.shape[3]
        if max(self.occ2map_prior_output_z_indices) >= z_dim:
            raise ValueError(
                'occ2map_prior_output_z_indices exceeds the decoded low-z '
                f'feature depth: indices={self.occ2map_prior_output_z_indices}, '
                f'z_dim={z_dim}.')

        z_indices = torch.tensor(
            self.occ2map_prior_output_z_indices,
            device=occ_logits.device,
            dtype=torch.long)
        selected = F.softmax(occ_logits.float(), dim=-1).index_select(
            3, z_indices)
        driveable_id, sidewalk_id = self.occ2map_prior_class_ids
        driveable_z = selected[..., driveable_id].clamp(0.0, 1.0)
        sidewalk_z = selected[..., sidewalk_id].clamp(0.0, 1.0)
        driveable = 1.0 - torch.prod(1.0 - driveable_z, dim=3)
        sidewalk = 1.0 - torch.prod(1.0 - sidewalk_z, dim=3)

        area_prior = torch.maximum(driveable, sidewalk)
        thin_prior = driveable
        prior = torch.stack((area_prior, thin_prior), dim=1).to(
            dtype=voxel_feature.dtype)

        grid = self._make_bev_grid(
            prior,
            self.occ2map_prior_feature_range,
            self.map_feature_range,
            self.map_feature_size)
        prior = F.grid_sample(
            prior,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        if self.occ2map_prior_detach:
            prior = prior.detach()
        return prior

    def _attach_occ2map_prior(self, map_feature, occ2map_prior):
        if occ2map_prior is None:
            return map_feature
        if isinstance(map_feature, dict):
            updated = dict(map_feature)
            updated['occ2map_prior'] = occ2map_prior
            return updated
        return dict(bev_feature=map_feature, occ2map_prior=occ2map_prior)

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

    def _check_map_feature_size(self, map_feature, gt_masks_bev):
        if gt_masks_bev is None:
            return
        map_tensor = self._get_map_feature_tensor(map_feature)
        if map_tensor.shape[-2:] != gt_masks_bev.shape[-2:]:
            raise ValueError(
                f'map feature size {map_tensor.shape[-2:]} does not match '
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
        voxel_feat, depth, pv_feat, map_lss2d_feat = \
            self.extract_feat_with_map_lss(
                img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        # Dual Branch Encoder (DBE)
        bev_branch_input = (
            map_lss2d_feat if self.use_lss2d_as_bev_branch else None)
        encoder_output = self.dual_branch_encoder(
            voxel_feat, bev_branch_input=bev_branch_input)
        comprehensive_voxel_feature, bev_feature, map_bev_feature = \
            self._split_encoder_output(encoder_output)
        occ_voxel_feature = self._crop_occ_feature(comprehensive_voxel_feature)
        occ_voxel_feature = self._apply_task_channel_scaling_occ(
            occ_voxel_feature)

        # 3d CNN for generating Prototype
        prototoype_occ_pred, mask_feat = self.cnn3d_decoder(occ_voxel_feature.permute(0,1,4,2,3))
            
        # Prototype Query Decoder (PQD)
        B = occ_voxel_feature.shape[0]
        img_metas = [{"pc_range": self.pc_range, "occ_size":self.grid_size} for i in range(B)]
        return_query_info = (
            self.bev_seg_head is not None and self._needs_pqd_query_info())
        pqd_outputs = self.prototype_query_decoder.forward_train(
            occ_voxel_feature,
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
            occ2map_prior = self._make_occ2map_prior(
                comprehensive_voxel_feature)
            lss2d_map_feature = self._encode_lss2d_map_feature(
                map_lss2d_feat)
            map_feature = self._prepare_map_feature(
                bev_feature, map_bev_feature, lss2d_map_feature)
            map_feature = self._apply_task_channel_scaling_map(map_feature)
            map_feature, ingest_losses = self._apply_voxel_aware_map_ingest(
                map_feature,
                occ_voxel_feature,
                query_info,
                gt_masks_bev=gt_masks_bev)
            map_feature = self._attach_occ2map_prior(
                map_feature, occ2map_prior)
            self._check_map_feature_size(map_feature, gt_masks_bev)
            bev_seg_logits = self.bev_seg_head(map_feature)
            map_losses = self.bev_seg_head.loss(bev_seg_logits, gt_masks_bev)
            losses.update(self._scale_map_losses(map_losses))
            losses.update(ingest_losses)

        if self.loss_uncertainty_weighting is not None:
            losses = self.loss_uncertainty_weighting(losses)

        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # 2D to 3D view transformation
        voxel_feat, depth, pv_feat, map_lss2d_feat = \
            self.extract_feat_with_map_lss(
                img_inputs=img, img_metas=img_metas, **kwargs)

        # Dual Branch Encoder (DBE)
        bev_branch_input = (
            map_lss2d_feat if self.use_lss2d_as_bev_branch else None)
        encoder_output = self.dual_branch_encoder(
            voxel_feat, bev_branch_input=bev_branch_input)
        comprehensive_voxel_feature, bev_feature, map_bev_feature = \
            self._split_encoder_output(encoder_output)
        occ_voxel_feature = self._crop_occ_feature(comprehensive_voxel_feature)
        occ_voxel_feature = self._apply_task_channel_scaling_occ(
            occ_voxel_feature)

        # Prototype Query Generator
        prototoype_occ_pred, mask_feat = self.cnn3d_decoder(occ_voxel_feature.permute(0,1,4,2,3))

        # Prototype Query Decoder (PQD)
        B = occ_voxel_feature.shape[0]
        img_metas_occ = [{"pc_range": self.pc_range, "occ_size":self.grid_size} for i in range(B)]
        return_query_info = (
            self.bev_seg_head is not None and self._needs_pqd_query_info())
        pqd_outputs = self.prototype_query_decoder.simple_test(
            occ_voxel_feature,
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

        lss2d_map_feature = self._encode_lss2d_map_feature(map_lss2d_feat)
        map_feature = self._prepare_map_feature(
            bev_feature, map_bev_feature, lss2d_map_feature)
        map_feature = self._apply_task_channel_scaling_map(map_feature)
        occ2map_prior = self._make_occ2map_prior(comprehensive_voxel_feature)
        map_feature, _ = self._apply_voxel_aware_map_ingest(
            map_feature, occ_voxel_feature, query_info)
        map_feature = self._attach_occ2map_prior(map_feature, occ2map_prior)
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
