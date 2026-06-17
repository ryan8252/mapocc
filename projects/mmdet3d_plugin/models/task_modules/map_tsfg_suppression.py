"""Map-TSFG supervised suppression for the BEV map branch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class MapTSFGSuppression(BaseModule):
    """MAESTRO-style map feature suppression before BEVSegHead.

    The suppress-only mode is intended as the clean first ablation: it keeps
    the current map feature path intact and learns a GT-supervised
    multiplicative gate on top of the 128-channel map neck output.
    """

    returns_aux_losses = True

    def __init__(self,
                 mode='full',
                 map_channels=128,
                 query_channels=48,
                 occ_num_classes=18,
                 background_class_ids=(11, 12, 13, 14, 15, 16),
                 prototype_source='query_norm_real_bqc',
                 detach_prototypes=False,
                 hidden_channels=128,
                 enhance_residual=True,
                 score_hidden_channels=64,
                 score_init_bias=2.0,
                 supp_loss_weight=1.0,
                 focal_gamma=2.0,
                 focal_alpha=-1.0,
                 gt_dilation=0,
                 loss_name='loss_map_supp',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if mode not in ('full', 'suppress_only'):
            raise ValueError(
                "mode must be 'full' or 'suppress_only', "
                f'got {mode!r}.')
        if prototype_source not in (
                'query_norm_real_bqc',
                'query_embed_real_bqc',
                'mask_embed_real_bqc'):
            raise ValueError(
                "prototype_source must be 'query_norm_real_bqc', "
                "'query_embed_real_bqc', or 'mask_embed_real_bqc'; got "
                f'{prototype_source!r}.')

        self.mode = mode
        self.map_channels = map_channels
        self.query_channels = query_channels
        self.occ_num_classes = occ_num_classes
        self.background_class_ids = list(background_class_ids)
        self.prototype_source = prototype_source
        self.detach_prototypes = bool(detach_prototypes)
        self.enhance_residual = bool(enhance_residual)
        self.supp_loss_weight = float(supp_loss_weight)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.gt_dilation = int(gt_dilation)
        self.loss_name = loss_name
        self.requires_pqd_query_info = (mode == 'full')

        if not self.background_class_ids:
            raise ValueError('background_class_ids cannot be empty.')
        if min(self.background_class_ids) < 0:
            raise ValueError('background_class_ids must be non-negative.')
        if max(self.background_class_ids) >= occ_num_classes:
            raise ValueError(
                f'background_class_ids {self.background_class_ids} exceed '
                f'occ_num_classes={occ_num_classes}.')
        if self.gt_dilation < 0:
            raise ValueError('gt_dilation must be >= 0.')

        num_bg = len(self.background_class_ids)
        if mode == 'full':
            self.prototype_proj = nn.Sequential(
                nn.Linear(query_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, map_channels),
            )
            self.prototype_gate = nn.Sequential(
                nn.Linear(2 * map_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, map_channels),
                nn.Sigmoid(),
            )
            self.enhance_conv = nn.Sequential(
                nn.Conv2d(map_channels + num_bg, map_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(map_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(map_channels, map_channels,
                          kernel_size=1, bias=True),
            )
            if self.enhance_residual:
                nn.init.zeros_(self.enhance_conv[-1].weight)
                nn.init.zeros_(self.enhance_conv[-1].bias)

        self.score_predictor = nn.Sequential(
            nn.Conv2d(map_channels, score_hidden_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(score_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(score_hidden_channels, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.score_predictor[-1].weight)
        nn.init.constant_(self.score_predictor[-1].bias, score_init_bias)

    def _get_bg_prototypes(self, query_info):
        query = query_info.get(self.prototype_source)
        if query is None:
            raise ValueError(
                f'{self.prototype_source} is required by MapTSFGSuppression.')
        if query.dim() != 3:
            raise ValueError(
                f'{self.prototype_source} must have shape [B, C_cls, C]; '
                f'got {tuple(query.shape)}.')
        if query.shape[1] < self.occ_num_classes:
            raise ValueError(
                f'{self.prototype_source} has {query.shape[1]} classes, '
                f'expected at least {self.occ_num_classes}.')
        if query.shape[2] != self.query_channels:
            raise ValueError(
                f'{self.prototype_source} has {query.shape[2]} channels, '
                f'expected {self.query_channels}.')

        query = query[:, self.background_class_ids, :]
        if self.detach_prototypes:
            query = query.detach()
        return self.prototype_proj(query)

    def _enhance(self, map_feature, prototypes):
        response = torch.einsum('bnc,bchw->bnhw', prototypes, map_feature)
        gamma = self.prototype_gate(
            torch.cat([
                prototypes.mean(dim=1),
                prototypes.max(dim=1).values,
            ], dim=-1))
        aware_feature = map_feature * gamma[:, :, None, None]
        delta = self.enhance_conv(torch.cat([response, aware_feature], dim=1))
        if self.enhance_residual:
            return map_feature + delta, aware_feature
        return delta, aware_feature

    def _make_supp_target(self, gt_masks_bev, like_logits):
        target = gt_masks_bev.float().amax(dim=1, keepdim=True)
        if self.gt_dilation > 0:
            kernel = 2 * self.gt_dilation + 1
            target = F.max_pool2d(
                target,
                kernel_size=kernel,
                stride=1,
                padding=self.gt_dilation)
        if target.shape[-2:] != like_logits.shape[-2:]:
            target = F.interpolate(
                target, size=like_logits.shape[-2:], mode='nearest')
        return target

    def _focal_loss(self, logits, target):
        logits = logits.float()
        target = target.float()
        prob = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction='none')
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        loss = ce_loss * ((1.0 - p_t) ** self.focal_gamma)
        if self.focal_alpha >= 0:
            alpha_t = (
                self.focal_alpha * target
                + (1.0 - self.focal_alpha) * (1.0 - target))
            loss = alpha_t * loss
        return loss.mean()

    def forward(self,
                map_feature,
                voxel_feature=None,
                gt_masks_bev=None,
                **query_info):
        if map_feature.dim() != 4:
            raise ValueError(
                'MapTSFGSuppression expects map_feature shape [B, C, H, W]; '
                f'got {tuple(map_feature.shape)}.')
        if map_feature.shape[1] != self.map_channels:
            raise ValueError(
                f'map_feature has {map_feature.shape[1]} channels, '
                f'expected {self.map_channels}.')

        if self.mode == 'full':
            f_tilde, score_source = self._enhance(
                map_feature, self._get_bg_prototypes(query_info))
        else:
            f_tilde = map_feature
            score_source = map_feature

        score_logits = self.score_predictor(score_source)
        output = f_tilde * torch.sigmoid(score_logits)

        losses = {}
        if self.training and gt_masks_bev is not None:
            target = self._make_supp_target(gt_masks_bev, score_logits)
            losses[self.loss_name] = (
                self.supp_loss_weight
                * self._focal_loss(score_logits, target))
        return output, losses
