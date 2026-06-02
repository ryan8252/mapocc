import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmdet3d.models.builder import HEADS, build_loss


@HEADS.register_module()
class BEVSegHead(BaseModule):
    """Lightweight BEV segmentation head for naive multi-task learning.

    This mirrors the common BEVFusion-style setup: shared BEV features are fed
    into a shallow 2D CNN decoder that predicts one binary mask per map class.
    """

    def __init__(self,
                 in_channels,
                 hidden_channels=128,
                 num_classes=6,
                 num_convs=2,
                 norm_cfg=dict(type='BN'),
                 with_cp=False,
                 loss_bce=None,
                 loss_dice=None,
                 loss_focal=None,
                 map_loss_balance_mode='none',
                 map_class_weights=None,
                 overlay_class_indices=None,
                 overlay_pos_weight=None,
                 dynamic_overlay_ref_pos_ratio=None,
                 dynamic_overlay_gamma=0.5,
                 dynamic_overlay_min_weight=1.0,
                 dynamic_overlay_max_weight=5.0,
                 dynamic_overlay_eps=1e-6,
                 map_balance_debug=False,
                 map_balance_debug_interval=50,
                 map_balance_class_names=None):
        super(BEVSegHead, self).__init__()
        self.with_cp = with_cp
        self.num_classes = num_classes
        self.map_loss_balance_mode = self._normalize_balance_mode(
            map_loss_balance_mode)
        self.map_class_weights = self._parse_class_weights(map_class_weights)
        self.overlay_class_indices = self._parse_overlay_indices(
            overlay_class_indices)
        self.overlay_pos_weight = self._parse_overlay_values(
            overlay_pos_weight, 'overlay_pos_weight', default=None)
        self.dynamic_overlay_ref_pos_ratio = self._parse_overlay_values(
            dynamic_overlay_ref_pos_ratio,
            'dynamic_overlay_ref_pos_ratio',
            default=None)
        self.dynamic_overlay_gamma = float(dynamic_overlay_gamma)
        self.dynamic_overlay_min_weight = float(dynamic_overlay_min_weight)
        self.dynamic_overlay_max_weight = float(dynamic_overlay_max_weight)
        self.dynamic_overlay_eps = float(dynamic_overlay_eps)
        self.map_balance_debug = bool(map_balance_debug)
        self.map_balance_debug_interval = int(map_balance_debug_interval)
        self.map_balance_class_names = self._parse_class_names(
            map_balance_class_names)
        self._map_balance_debug_step = 0
        self._validate_balance_cfg()

        blocks = []
        current_channels = in_channels
        for _ in range(num_convs):
            blocks.append(
                ConvModule(
                    current_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU', inplace=True),
                ))
            current_channels = hidden_channels
        self.decoder = nn.Sequential(*blocks)
        self.predictor = nn.Conv2d(current_channels, num_classes, kernel_size=1)

        self.loss_bce = build_loss(loss_bce) if loss_bce is not None else None
        self.loss_dice = build_loss(loss_dice) if loss_dice is not None else None
        self.loss_focal = build_loss(loss_focal) if loss_focal is not None else None

    def _normalize_balance_mode(self, mode):
        if mode is None:
            return 'none'
        mode = str(mode).lower()
        aliases = {
            'off': 'none',
            'disable': 'none',
            'disabled': 'none',
            'static': 'class_static',
            'overlay': 'overlay_static',
            'dynamic': 'overlay_dynamic',
            'v3': 'overlay_dynamic',
        }
        mode = aliases.get(mode, mode)
        valid_modes = {'none', 'class_static', 'overlay_static',
                       'overlay_dynamic'}
        if mode not in valid_modes:
            raise ValueError(
                f'Unsupported map_loss_balance_mode: {mode}. '
                f'Expected one of {sorted(valid_modes)}.')
        return mode

    def _parse_class_weights(self, weights):
        if weights is None:
            return None
        weights = [float(weight) for weight in weights]
        if len(weights) != self.num_classes:
            raise ValueError(
                'map_class_weights length must match num_classes: '
                f'{len(weights)} vs {self.num_classes}.')
        return weights

    def _parse_overlay_indices(self, indices):
        if indices is None:
            return []
        indices = [int(index) for index in indices]
        for index in indices:
            if index < 0 or index >= self.num_classes:
                raise ValueError(
                    'overlay_class_indices contains an out-of-range index: '
                    f'{index} for num_classes={self.num_classes}.')
        if len(set(indices)) != len(indices):
            raise ValueError('overlay_class_indices must not contain duplicates.')
        return indices

    def _parse_overlay_values(self, values, name, default=None):
        if not self.overlay_class_indices:
            if values is not None:
                raise ValueError(
                    f'{name} requires overlay_class_indices to be set.')
            return None
        if values is None:
            if default is None:
                return None
            return [float(default) for _ in self.overlay_class_indices]
        if isinstance(values, (int, float)):
            return [float(values) for _ in self.overlay_class_indices]

        values = [float(value) for value in values]
        if len(values) == len(self.overlay_class_indices):
            return values
        if len(values) == self.num_classes:
            return [values[index] for index in self.overlay_class_indices]
        raise ValueError(
            f'{name} must be a scalar, have length '
            f'{len(self.overlay_class_indices)} for overlay classes, or have '
            f'length {self.num_classes} for all map classes.')

    def _parse_class_names(self, class_names):
        if class_names is None:
            return [f'class_{index}' for index in range(self.num_classes)]
        class_names = list(class_names)
        if len(class_names) != self.num_classes:
            raise ValueError(
                'map_balance_class_names length must match num_classes: '
                f'{len(class_names)} vs {self.num_classes}.')
        return class_names

    def _validate_balance_cfg(self):
        mode = self.map_loss_balance_mode
        if mode == 'class_static' and self.map_class_weights is None:
            raise ValueError(
                'class_static map balancing requires map_class_weights.')
        if mode == 'overlay_static':
            if not self.overlay_class_indices:
                raise ValueError(
                    'overlay_static map balancing requires '
                    'overlay_class_indices.')
            if self.overlay_pos_weight is None:
                raise ValueError(
                    'overlay_static map balancing requires overlay_pos_weight.')
        if mode == 'overlay_dynamic':
            if not self.overlay_class_indices:
                raise ValueError(
                    'overlay_dynamic map balancing requires '
                    'overlay_class_indices.')
            if self.dynamic_overlay_ref_pos_ratio is None:
                raise ValueError(
                    'overlay_dynamic map balancing requires '
                    'dynamic_overlay_ref_pos_ratio.')
            if any(ratio <= 0 for ratio in self.dynamic_overlay_ref_pos_ratio):
                raise ValueError(
                    'dynamic_overlay_ref_pos_ratio values must be positive.')

        if self.dynamic_overlay_gamma < 0:
            raise ValueError('dynamic_overlay_gamma must be non-negative.')
        if self.dynamic_overlay_min_weight <= 0:
            raise ValueError(
                'dynamic_overlay_min_weight must be greater than zero.')
        if self.dynamic_overlay_max_weight < self.dynamic_overlay_min_weight:
            raise ValueError(
                'dynamic_overlay_max_weight must be >= '
                'dynamic_overlay_min_weight.')
        if self.dynamic_overlay_eps <= 0:
            raise ValueError('dynamic_overlay_eps must be greater than zero.')

    def forward(self, bev_feature):
        if self.with_cp and self.training:
            bev_feature = checkpoint(self.decoder, bev_feature)
        else:
            bev_feature = self.decoder(bev_feature)
        return self.predictor(bev_feature)

    def loss(self, seg_logits, gt_masks_bev):
        gt_masks_bev = gt_masks_bev.float()
        if self.map_loss_balance_mode != 'none':
            return self._balanced_loss(seg_logits, gt_masks_bev)

        losses = {}

        if self.loss_bce is not None:
            losses['loss_map_bce'] = self.loss_bce(seg_logits, gt_masks_bev)

        if self.loss_focal is not None:
            losses['loss_map_focal'] = self.loss_focal(seg_logits, gt_masks_bev)

        if self.loss_dice is not None:
            losses['loss_map_dice'] = self.loss_dice(
                seg_logits.reshape(-1, *seg_logits.shape[2:]),
                gt_masks_bev.reshape(-1, *gt_masks_bev.shape[2:]),
            )

        return losses

    def _balanced_loss(self, seg_logits, gt_masks_bev):
        if seg_logits.shape != gt_masks_bev.shape:
            raise ValueError(
                'BEVSegHead balanced loss expects seg_logits and '
                'gt_masks_bev to have the same shape, but got '
                f'{tuple(seg_logits.shape)} and {tuple(gt_masks_bev.shape)}.')

        losses = {}
        bce_weight, dice_weight, balance_info = self._build_balance_weights(
            gt_masks_bev)

        bce_contrib = None
        if self.loss_bce is not None:
            if not getattr(self.loss_bce, 'use_sigmoid', True):
                raise NotImplementedError(
                    'BEVSegHead balanced BCE currently supports only '
                    'sigmoid BCE.')
            bce_per_pixel = F.binary_cross_entropy_with_logits(
                seg_logits, gt_masks_bev, reduction='none')
            bce_weighted = bce_per_pixel * bce_weight
            bce_loss_weight = float(getattr(self.loss_bce, 'loss_weight', 1.0))
            losses['loss_map_bce'] = bce_loss_weight * bce_weighted.mean()
            bce_contrib = bce_loss_weight * bce_weighted.mean(dim=(0, 2, 3))

        dice_contrib = None
        if self.loss_dice is not None:
            dice_per_class = self._dice_loss_per_class(
                seg_logits, gt_masks_bev)
            dice_weighted = dice_per_class * dice_weight
            dice_loss_weight = float(
                getattr(self.loss_dice, 'loss_weight', 1.0))
            losses['loss_map_dice'] = dice_loss_weight * dice_weighted.mean()
            dice_contrib = dice_loss_weight * dice_weighted.mean(dim=0)

        self._log_balance_debug(balance_info, bce_contrib, dice_contrib)
        return losses

    def _build_balance_weights(self, gt_masks_bev):
        bce_weight = torch.ones_like(gt_masks_bev)
        dice_weight = gt_masks_bev.new_ones(
            (gt_masks_bev.shape[0], gt_masks_bev.shape[1]))
        pos_pixels = gt_masks_bev.sum(dim=(0, 2, 3))
        valid_pixels = max(
            int(gt_masks_bev.shape[0] * gt_masks_bev.shape[2]
                * gt_masks_bev.shape[3]), 1)
        pos_ratio = pos_pixels / float(valid_pixels)

        class_weights = None
        overlay_weights_full = gt_masks_bev.new_ones(self.num_classes)
        ref_ratio_full = gt_masks_bev.new_zeros(self.num_classes)
        present_full = pos_pixels > 0

        if self.map_loss_balance_mode == 'class_static':
            class_weights = gt_masks_bev.new_tensor(self.map_class_weights)
            bce_weight = bce_weight * class_weights.view(1, -1, 1, 1)
            dice_weight = dice_weight * class_weights.view(1, -1)

        if self.map_loss_balance_mode in ('overlay_static',
                                          'overlay_dynamic'):
            overlay_indices = torch.tensor(
                self.overlay_class_indices,
                device=gt_masks_bev.device,
                dtype=torch.long)
            present_overlay = pos_pixels[overlay_indices] > 0

            if self.map_loss_balance_mode == 'overlay_static':
                overlay_weights = gt_masks_bev.new_tensor(
                    self.overlay_pos_weight)
            else:
                ref_ratio = gt_masks_bev.new_tensor(
                    self.dynamic_overlay_ref_pos_ratio)
                batch_ratio = pos_ratio[overlay_indices]
                overlay_weights = (
                    ref_ratio / (batch_ratio + self.dynamic_overlay_eps)
                ).pow(self.dynamic_overlay_gamma)
                overlay_weights = torch.clamp(
                    overlay_weights,
                    min=self.dynamic_overlay_min_weight,
                    max=self.dynamic_overlay_max_weight)
                ref_ratio_full[overlay_indices] = ref_ratio

            overlay_weights = torch.where(
                present_overlay,
                overlay_weights,
                torch.ones_like(overlay_weights))
            overlay_weights_full[overlay_indices] = overlay_weights

            for offset, class_index in enumerate(self.overlay_class_indices):
                if bool(present_overlay[offset].item()):
                    weight = overlay_weights[offset]
                    bce_weight[:, class_index] = (
                        1.0 + (weight - 1.0) * gt_masks_bev[:, class_index])
                    dice_weight[:, class_index] = weight

        balance_info = dict(
            mode=self.map_loss_balance_mode,
            pos_pixels=pos_pixels.detach(),
            pos_ratio=pos_ratio.detach(),
            class_weights=class_weights.detach()
            if class_weights is not None else None,
            overlay_weights=overlay_weights_full.detach(),
            ref_ratio=ref_ratio_full.detach(),
            present=present_full.detach())
        return bce_weight, dice_weight, balance_info

    def _dice_loss_per_class(self, seg_logits, gt_masks_bev):
        pred = seg_logits
        if getattr(self.loss_dice, 'activate', True):
            if getattr(self.loss_dice, 'use_sigmoid', True):
                pred = pred.sigmoid()
            else:
                raise NotImplementedError(
                    'BEVSegHead balanced Dice currently supports only '
                    'sigmoid activation.')

        pred = pred.flatten(2)
        target = gt_masks_bev.flatten(2).float()
        eps = float(getattr(self.loss_dice, 'eps', 1e-3))
        intersection = torch.sum(pred * target, dim=2)

        if getattr(self.loss_dice, 'naive_dice', False):
            denominator = torch.sum(pred, dim=2) + torch.sum(target, dim=2)
            dice_score = (2 * intersection + eps) / (denominator + eps)
        else:
            pred_area = torch.sum(pred * pred, dim=2) + eps
            target_area = torch.sum(target * target, dim=2) + eps
            dice_score = (2 * intersection) / (pred_area + target_area)

        return 1 - dice_score

    def _format_class_values(self, values, precision=4):
        if values is None:
            return 'none'
        values = values.detach().cpu().tolist()
        parts = []
        for name, value in zip(self.map_balance_class_names, values):
            if isinstance(value, bool):
                formatted = str(value)
            elif float(value).is_integer():
                formatted = str(int(value))
            else:
                formatted = f'{float(value):.{precision}g}'
            parts.append(f'{name}:{formatted}')
        return ', '.join(parts)

    def _log_balance_debug(self, balance_info, bce_contrib, dice_contrib):
        if not self.map_balance_debug:
            return
        step = self._map_balance_debug_step
        self._map_balance_debug_step += 1
        if (self.map_balance_debug_interval > 1
                and step % self.map_balance_debug_interval != 0):
            return

        logger = logging.getLogger('mmdet')
        logger.info(
            '[BEVSegHead map balance] '
            f'step={step} mode={balance_info["mode"]} '
            'pos_pixels=('
            f'{self._format_class_values(balance_info["pos_pixels"], 0)}) '
            'pos_ratio=('
            f'{self._format_class_values(balance_info["pos_ratio"], 6)}) '
            'weights=('
            f'{self._format_class_values(balance_info["overlay_weights"], 4)}) '
            'ref_ratio=('
            f'{self._format_class_values(balance_info["ref_ratio"], 6)}) '
            'bce_contrib=('
            f'{self._format_class_values(bce_contrib, 4)}) '
            'dice_contrib=('
            f'{self._format_class_values(dice_contrib, 4)})')

    def predict(self, seg_logits):
        return torch.sigmoid(seg_logits)
