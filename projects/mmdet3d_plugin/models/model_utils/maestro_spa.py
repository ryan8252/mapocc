import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from mmdet3d.models import BACKBONES


@BACKBONES.register_module()
class MAESTROOnewayScenePrototypeAggregator(BaseModule):
    """One-way SPA for occupancy + map segmentation.

    The original MAESTRO SPA aggregates detection and map task prototypes into
    occupancy prototypes. This 2-task adaptation only lets the map branch enrich
    occupancy prototypes. The map source can be detached so occupancy gradients
    do not rewrite the map-specific branch.
    """

    def __init__(self,
                 prototype_channels,
                 map_feature_channels=None,
                 num_map_classes=6,
                 detach_source=True,
                 init_cfg=None):
        super(MAESTROOnewayScenePrototypeAggregator, self).__init__(init_cfg)
        map_feature_channels = map_feature_channels or prototype_channels
        self.prototype_channels = prototype_channels
        self.map_feature_channels = map_feature_channels
        self.num_map_classes = num_map_classes
        self.detach_source = detach_source

        self.map_proto_proj = nn.Sequential(
            nn.Linear(map_feature_channels, prototype_channels),
            nn.ReLU(inplace=True),
            nn.Linear(prototype_channels, prototype_channels),
        )
        self.fusion_scale = nn.Parameter(torch.zeros(1))

    def _pool_map_prototypes(self, map_feature, map_logits):
        if self.detach_source:
            map_feature = map_feature.detach()
            map_logits = map_logits.detach()

        weights = torch.sigmoid(map_logits)
        feat_flat = map_feature.flatten(2)
        weight_flat = weights.flatten(2)
        denom = weight_flat.sum(dim=-1, keepdim=True).clamp_min(1.0)
        map_prototypes = torch.einsum('bks,bcs->bkc', weight_flat, feat_flat)
        map_prototypes = map_prototypes / denom
        return self.map_proto_proj(map_prototypes)

    def forward(self, background_prototypes, map_feature, map_logits):
        if background_prototypes is None or background_prototypes.size(1) == 0:
            return background_prototypes

        map_prototypes = self._pool_map_prototypes(map_feature, map_logits)
        if map_prototypes.size(1) == background_prototypes.size(1):
            aligned = map_prototypes
        else:
            bg_norm = F.normalize(background_prototypes, dim=-1)
            map_norm = F.normalize(map_prototypes, dim=-1)
            attn = torch.einsum('bnc,bkc->bnk', bg_norm, map_norm)
            attn = F.softmax(attn, dim=-1)
            aligned = torch.einsum('bnk,bkc->bnc', attn, map_prototypes)

        return background_prototypes + self.fusion_scale * aligned
