import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import ModuleList, force_fp32
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.builder import HEADS, build_loss

from .base.anchor_free_head import AnchorFreeHead
from .base.mmdet_utils import point_sample_3d, preprocess_occupancy_gt


@HEADS.register_module()
class MAESTROOccFormerHead(AnchorFreeHead):
    """OccFormer-style Mask2Former head for ProtoOcc-format occupancy labels.

    The original OccFormer head consumes a multi-scale 3D feature pyramid and
    lidarseg point labels. This adapter builds the pyramid inside the head from
    the MAESTRO occupancy feature and supervises masks with voxel_semantics plus
    mask_camera, so it can train inside the existing ProtoOcc multitask dataset.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_occupancy_classes=18,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 transformer_decoder=None,
                 positional_encoding=None,
                 pooling_attn_mask=True,
                 padding_mode='border',
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 norm_cfg=dict(type='BN3d'),
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.num_occupancy_classes = num_occupancy_classes
        self.num_classes = num_occupancy_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.pooling_attn_mask = pooling_attn_mask
        self.padding_mode = padding_mode
        self.align_corners = True

        self.input_proj = ConvModule(
            in_channels,
            feat_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True),
        )
        self.mask_feature_proj = nn.Conv3d(
            feat_channels, out_channels, kernel_size=1)

        self.pyramid_blocks = nn.ModuleList()
        for _ in range(num_transformer_feat_level):
            self.pyramid_blocks.append(
                ConvModule(
                    feat_channels,
                    feat_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU', inplace=True),
                ))

        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers

        self.decoder_input_projs = ModuleList()
        for _ in range(num_transformer_feat_level):
            if self.decoder_embed_dims != feat_channels:
                self.decoder_input_projs.append(
                    nn.Conv3d(feat_channels, self.decoder_embed_dims, 1))
            else:
                self.decoder_input_projs.append(nn.Identity())

        self.decoder_positional_encoding = build_positional_encoding(
            positional_encoding)
        self.query_embed = nn.Embedding(num_queries, feat_channels)
        self.query_feat = nn.Embedding(num_queries, feat_channels)
        self.level_embed = nn.Embedding(num_transformer_feat_level,
                                        feat_channels)

        self.cls_embed = nn.Linear(feat_channels, num_occupancy_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels),
        )

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(train_cfg.get('assigner', None))
            self.sampler = build_sampler(
                train_cfg.get('sampler', None), context=self)
            self.num_points = train_cfg.get('num_points', 12544)
            self.oversample_ratio = train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.preprocess_occupancy_gt = preprocess_occupancy_gt
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        self.class_weight = loss_cls.get(
            'class_weight', [1.0] * num_occupancy_classes + [0.1])

    def init_weights(self):
        for module in self.decoder_input_projs:
            if isinstance(module, nn.Conv3d):
                caffe2_xavier_init(module, bias=0)
        for param in self.transformer_decoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    def _build_pyramid(self, voxel_feature):
        base_feature = self.input_proj(voxel_feature)
        mask_feature = self.mask_feature_proj(base_feature)
        pyramid = [mask_feature]

        memory = base_feature
        for block in self.pyramid_blocks:
            memory = block(memory)
            pyramid.append(memory)
        return pyramid

    def _sample_points(self, gt_binary, mask_camera, num_points):
        valid_mask = torch.ones_like(gt_binary, dtype=torch.bool)
        if mask_camera is not None:
            valid_mask = mask_camera.bool()

        positive = gt_binary.bool() & valid_mask
        positive_coords = positive.nonzero(as_tuple=False)
        valid_coords = valid_mask.nonzero(as_tuple=False)
        coords = []

        num_positive = min(num_points // 2, positive_coords.shape[0])
        if num_positive > 0:
            perm = torch.randperm(
                positive_coords.shape[0], device=positive_coords.device)
            coords.append(positive_coords[perm[:num_positive]])

        remaining = num_points - num_positive
        if remaining > 0 and valid_coords.shape[0] > 0:
            take_valid = min(remaining, valid_coords.shape[0])
            perm = torch.randperm(valid_coords.shape[0],
                                  device=valid_coords.device)
            coords.append(valid_coords[perm[:take_valid]])
            remaining -= take_valid

        if remaining > 0:
            rand_coords = torch.rand(
                (remaining, 3),
                device=gt_binary.device,
                dtype=torch.float32,
            )
            if coords:
                coords = torch.cat(coords, dim=0).float()
                spatial = torch.tensor(
                    gt_binary.shape,
                    device=gt_binary.device,
                    dtype=torch.float32,
                ).clamp_min(2.0) - 1.0
                coords = coords / spatial.view(1, 3)
                return torch.cat([coords, rand_coords], dim=0)
            return rand_coords

        coords = torch.cat(coords, dim=0).float()
        spatial = torch.tensor(
            gt_binary.shape,
            device=gt_binary.device,
            dtype=torch.float32,
        ).clamp_min(2.0) - 1.0
        return coords / spatial.view(1, 3)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks,
                           gt_binary, mask_camera, img_meta):
        num_queries = cls_score.shape[0]
        gt_labels = gt_labels.long()
        num_gts = gt_labels.shape[0]

        labels = gt_labels.new_full((num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = cls_score.new_ones(num_queries)
        mask_weights = mask_pred.new_zeros((num_queries, ))

        if num_gts == 0:
            pos_inds = torch.empty(0, dtype=torch.long, device=mask_pred.device)
            neg_inds = torch.arange(num_queries, device=mask_pred.device)
            empty_masks = gt_masks.new_empty((0, ) + gt_masks.shape[1:])
            return (labels, label_weights, empty_masks, mask_weights,
                    pos_inds, neg_inds)

        point_coords = self._sample_points(
            gt_binary, mask_camera, self.num_points)
        pred_points = point_coords.repeat(num_queries, 1, 1)[..., [2, 1, 0]]
        gt_points = point_coords.repeat(num_gts, 1, 1)[..., [2, 1, 0]]

        mask_points_pred = point_sample_3d(
            mask_pred.unsqueeze(1),
            pred_points,
            align_corners=self.align_corners,
            padding_mode=self.padding_mode,
        ).squeeze(1)
        gt_points_masks = point_sample_3d(
            gt_masks.unsqueeze(1).float(),
            gt_points,
            align_corners=self.align_corners,
            padding_mode=self.padding_mode,
        ).squeeze(1)

        assign_result = self.assigner.assign(
            cls_score, mask_points_pred, gt_labels, gt_points_masks, img_meta)
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        class_weights = cls_score.new_tensor(self.class_weight)
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights[pos_inds] = class_weights[labels[pos_inds]]

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, gt_binary_list, mask_camera_list,
                    img_metas):
        results = multi_apply(
            self._get_target_single,
            cls_scores_list,
            mask_preds_list,
            gt_labels_list,
            gt_masks_list,
            gt_binary_list,
            mask_camera_list,
            img_metas,
        )
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list, neg_inds_list) = results
        num_total_pos = sum(inds.numel() for inds in pos_inds_list)
        num_total_neg = sum(inds.numel() for inds in neg_inds_list)
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list,
             gt_masks_list, gt_binary_list, mask_camera, img_metas):
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_binary_list = [gt_binary_list for _ in range(num_dec_layers)]
        mask_camera_list = [
            [None] * len(gt_labels_list)
            if mask_camera is None else [mask_camera[i] for i in range(len(gt_labels_list))]
            for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_mask_preds,
            all_gt_labels_list,
            all_gt_masks_list,
            all_gt_binary_list,
            mask_camera_list,
            img_metas_list,
        )

        loss_dict = dict(
            loss_occformer_cls=losses_cls[-1],
            loss_occformer_mask=losses_mask[-1],
            loss_occformer_dice=losses_dice[-1],
        )
        for layer_index, (loss_cls, loss_mask, loss_dice) in enumerate(
                zip(losses_cls[:-1], losses_mask[:-1], losses_dice[:-1])):
            loss_dict[f'd{layer_index}.loss_occformer_cls'] = loss_cls
            loss_dict[f'd{layer_index}.loss_occformer_mask'] = loss_mask
            loss_dict[f'd{layer_index}.loss_occformer_dice'] = loss_dice
        return loss_dict

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, gt_binary_list, mask_camera_list,
                    img_metas):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        targets = self.get_targets(
            cls_scores_list,
            mask_preds_list,
            gt_labels_list,
            gt_masks_list,
            gt_binary_list,
            mask_camera_list,
            img_metas,
        )
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos, _) = targets

        labels = torch.stack(labels_list, dim=0)
        label_weights = torch.stack(label_weights_list, dim=0)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores.flatten(0, 1),
            labels.flatten(0, 1),
            label_weights.flatten(0, 1),
            avg_factor=class_weight[labels.flatten(0, 1)].sum(),
        )

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = torch.clamp(num_total_masks, min=1.0)
        loss_mask = cls_scores.new_tensor(0.0)
        loss_dice = cls_scores.new_tensor(0.0)

        for batch_index in range(num_imgs):
            pos_mask = mask_weights[batch_index] > 0
            if pos_mask.sum() == 0:
                continue

            pos_pred = mask_preds[batch_index][pos_mask]
            pos_target = mask_targets_list[batch_index]
            pos_weight = mask_weights[batch_index][pos_mask]

            point_coords = self._sample_points(
                gt_binary_list[batch_index],
                mask_camera_list[batch_index],
                self.num_points,
            )
            pred_points = point_coords.repeat(pos_pred.shape[0], 1,
                                              1)[..., [2, 1, 0]]
            target_points = point_coords.repeat(pos_target.shape[0], 1,
                                                1)[..., [2, 1, 0]]

            mask_point_preds = point_sample_3d(
                pos_pred.unsqueeze(1),
                pred_points,
                align_corners=self.align_corners,
                padding_mode=self.padding_mode,
            ).squeeze(1)
            mask_point_targets = point_sample_3d(
                pos_target.unsqueeze(1).float(),
                target_points,
                align_corners=self.align_corners,
                padding_mode=self.padding_mode,
            ).squeeze(1)

            avg_factor = torch.clamp(
                reduce_mean(pos_weight.sum()), min=1.0)
            loss_dice = loss_dice + self.loss_dice(
                mask_point_preds,
                mask_point_targets,
                weight=pos_weight,
                avg_factor=avg_factor,
            )
            loss_mask = loss_mask + self.loss_mask(
                mask_point_preds.reshape(-1),
                mask_point_targets.reshape(-1),
                avg_factor=avg_factor * self.num_points,
            )

        return loss_cls, loss_mask, loss_dice

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        cls_pred = self.cls_embed(decoder_out)
        mask_embed = self.mask_embed(decoder_out)
        mask_pred = torch.einsum('bqc,bcxyz->bqxyz', mask_embed, mask_feature)

        if self.pooling_attn_mask:
            attn_mask = F.adaptive_max_pool3d(mask_pred.float(),
                                              attn_mask_target_size)
        else:
            attn_mask = F.interpolate(
                mask_pred,
                attn_mask_target_size,
                mode='trilinear',
                align_corners=self.align_corners,
            )
        attn_mask = attn_mask.flatten(2).detach().sigmoid() < 0.5
        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1).flatten(0, 1)
        return cls_pred, mask_pred, attn_mask

    def preprocess_gt(self, gt_occ, img_metas):
        num_class_list = [self.num_occupancy_classes] * len(img_metas)
        labels, masks, binary = multi_apply(
            self.preprocess_occupancy_gt, gt_occ, num_class_list)
        return labels, masks, binary

    def forward_train(self,
                      voxel_feats,
                      img_metas,
                      gt_occ,
                      mask_camera=None,
                      **kwargs):
        all_cls_scores, all_mask_preds = self(voxel_feats, img_metas)
        gt_labels, gt_masks, gt_binary = self.preprocess_gt(gt_occ, img_metas)
        return self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks,
                         gt_binary, mask_camera, img_metas)

    def forward(self, voxel_feats, img_metas, **kwargs):
        batch_size = len(img_metas)
        if isinstance(voxel_feats, torch.Tensor):
            voxel_feats = self._build_pyramid(voxel_feats)

        mask_features = voxel_feats[0]
        multi_scale_memorys = voxel_feats[:0:-1]

        decoder_inputs = []
        decoder_positional_encodings = []
        for level_index in range(self.num_transformer_feat_level):
            memory = multi_scale_memorys[level_index]
            decoder_input = self.decoder_input_projs[level_index](memory)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[level_index].view(1, 1, -1)
            decoder_input = decoder_input + level_embed

            mask = decoder_input.new_zeros(
                (batch_size, ) + memory.shape[-3:], dtype=torch.bool)
            decoder_pos = self.decoder_positional_encoding(mask)
            decoder_pos = decoder_pos.flatten(2).permute(2, 0, 1)

            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_pos)

        query_feat = self.query_feat.weight.unsqueeze(1).repeat(
            1, batch_size, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            1, batch_size, 1)

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self.forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-3:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for layer_index in range(self.num_transformer_decoder_layers):
            level_index = layer_index % self.num_transformer_feat_level
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            layer = self.transformer_decoder.layers[layer_index]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_index],
                value=decoder_inputs[level_index],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_index],
                attn_masks=[attn_mask, None],
                query_key_padding_mask=None,
                key_padding_mask=None,
            )
            next_level = (layer_index + 1) % self.num_transformer_feat_level
            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[next_level].shape[-3:],
            )
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list

    def format_results(self, mask_cls_results, mask_pred_results):
        mask_cls = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        return torch.einsum('bqc,bqxyz->bcxyz', mask_cls, mask_pred)

    def simple_test(self, voxel_feats, img_metas, **kwargs):
        all_cls_scores, all_mask_preds = self(voxel_feats, img_metas)
        output_voxels = self.format_results(all_cls_scores[-1],
                                            all_mask_preds[-1])

        target_size = img_metas[0].get('occ_size')
        if target_size is not None and tuple(output_voxels.shape[-3:]) != tuple(target_size):
            output_voxels = F.interpolate(
                output_voxels,
                size=tuple(target_size),
                mode='trilinear',
                align_corners=self.align_corners,
            )

        occ_pred = output_voxels.argmax(dim=1)
        return list(occ_pred.detach().cpu().numpy().astype(np.uint8))
