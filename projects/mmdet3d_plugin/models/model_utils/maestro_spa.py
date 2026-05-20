import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmdet3d.models import BACKBONES


@BACKBONES.register_module()
class MAESTROOnewayScenePrototypeAggregator(BaseModule):
    """One-way SPA for occupancy + map segmentation.

    Paper mapping:
    - Task-oriented Prototype Generation: P_map from map masks and map feature
    - Scene Prototype Aggregation: P_scene = P_bg + A(P_map)

    The original MAESTRO SPA aggregates both detection and map task-oriented
    prototypes into occupancy prototypes. This repo is the requested 2-task
    reproduction, so the available SPA path is one-way map-to-occ only. The
    aggregation follows MAESTRO's explicit semantic rules: one-to-one labels are
    directly summed; multiple fine-grained labels are averaged before summation.
    """

    def __init__(self,
                 prototype_channels,
                 map_feature_channels=None,
                 num_map_classes=6,
                 detach_source=True,
                 aggregation_weight=1.0,
                 semantic_aggregation_rules=None,
                 init_cfg=None):
        super(MAESTROOnewayScenePrototypeAggregator, self).__init__(init_cfg)
        map_feature_channels = map_feature_channels or prototype_channels
        self.prototype_channels = prototype_channels
        self.map_feature_channels = map_feature_channels
        self.num_map_classes = num_map_classes
        self.detach_source = detach_source
        self.aggregation_weight = aggregation_weight
        self.semantic_aggregation_rules = self._normalize_rules(
            semantic_aggregation_rules)

        self.map_prototype_projector = nn.Sequential(
            nn.Linear(map_feature_channels, prototype_channels),
            nn.ReLU(inplace=True),
            nn.Linear(prototype_channels, prototype_channels),
        )

    def _normalize_rules(self, semantic_aggregation_rules):
        if semantic_aggregation_rules is None:
            # Default nuScenes 2-task rule set.
            # P_bg order: driveable_surface, other_flat, sidewalk, terrain,
            # manmade, vegetation.
            # P_map order: drivable_area, ped_crossing, walkway, stop_line,
            # carpark_area, divider.
            semantic_aggregation_rules = (
                (0, (0, 1, 3, 5)),
                (1, (4,)),
                (2, (2,)),
            )

        if isinstance(semantic_aggregation_rules, dict):
            rule_iter = semantic_aggregation_rules.items()
        else:
            rule_iter = semantic_aggregation_rules

        normalized_rules = []
        for background_index, map_indices in rule_iter:
            if isinstance(map_indices, int):
                map_indices = (map_indices,)
            normalized_rules.append(
                (int(background_index),
                 tuple(int(map_index) for map_index in map_indices)))
        return tuple(normalized_rules)

    def _pool_map_prototypes(self, map_feature, map_logits):
        if self.detach_source:
            map_feature = map_feature.detach()
            map_logits = map_logits.detach()

        # Task-oriented Map Prototype Generation: P_map.
        weights = torch.sigmoid(map_logits)
        feat_flat = map_feature.flatten(2)
        weight_flat = weights.flatten(2)
        # Masked Average Pooling
        denom = weight_flat.sum(dim=-1, keepdim=True).clamp_min(1.0)
        P_map = torch.einsum('bks,bcs->bkc', weight_flat, feat_flat)
        P_map = P_map / denom
        return self.map_prototype_projector(P_map)

    def _aggregate_by_semantic_rules(self, background_prototypes, map_prototypes):
        num_background = background_prototypes.size(1)
        num_map = map_prototypes.size(1)
        aggregated_P_map = background_prototypes.new_zeros(
            background_prototypes.shape)

        for background_index, map_indices in self.semantic_aggregation_rules:
            if background_index < 0 or background_index >= num_background:
                raise ValueError(
                    f'Background prototype index {background_index} is out of '
                    f'range for {num_background} background prototypes.')
            if len(map_indices) == 0:
                continue
            if min(map_indices) < 0 or max(map_indices) >= num_map:
                raise ValueError(
                    f'Map prototype indices {map_indices} are out of range '
                    f'for {num_map} map prototypes.')

            map_indices_tensor = map_prototypes.new_tensor(
                map_indices, dtype=torch.long)
            selected_P_map = map_prototypes.index_select(1, map_indices_tensor)

            # Prototype aggregation: multiple fine-grained map labels are
            # averaged first, then summed with the corresponding P_bg.
            aggregated_P_map[:, background_index] = (
                aggregated_P_map[:, background_index] +
                selected_P_map.mean(dim=1))

        return aggregated_P_map

    def forward(self, background_prototypes, map_feature, map_logits):
        if background_prototypes is None or background_prototypes.size(1) == 0:
            return background_prototypes

        # Task-oriented prototype generation
        map_prototypes = self._pool_map_prototypes(map_feature, map_logits)
        # Prototype aggregation
        aggregated_P_map = self._aggregate_by_semantic_rules(background_prototypes, map_prototypes)

        # Scene Prototype Aggregation: P_scene = P_bg + A(P_map).
        return background_prototypes + self.aggregation_weight * aggregated_P_map
