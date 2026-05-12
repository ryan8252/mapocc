import torch
from torch import nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class OccToMapGeometryAdapter(BaseModule):
    """Convert coarse occupancy logits into a BEV map-logit residual."""

    def __init__(self,
                 num_occ_classes=18,
                 num_map_classes=6,
                 hidden_channels=64,
                 num_convs=2,
                 beta_o2m_max=1.0,
                 zero_until_epoch=6,
                 warmup_epochs=4,
                 detach_occ=True,
                 free_class_index=17,
                 ground_layout_indices=(11, 12, 13, 14),
                 structure_indices=(15, 16),
                 dynamic_indices=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                 height_threshold=0.5,
                 eps=1e-6,
                 norm_cfg=dict(type='BN'),
                 with_cp=False,
                 zero_init_output=True,
                 debug=False):
        super(OccToMapGeometryAdapter, self).__init__()
        self.num_occ_classes = int(num_occ_classes)
        self.num_map_classes = int(num_map_classes)
        self.beta_o2m_max = float(beta_o2m_max)
        self.zero_until_epoch = int(zero_until_epoch)
        self.warmup_epochs = int(warmup_epochs)
        self.detach_occ = bool(detach_occ)
        self.free_class_index = int(free_class_index)
        self.height_threshold = float(height_threshold)
        self.eps = float(eps)
        self.with_cp = bool(with_cp)
        self.zero_init_output = bool(zero_init_output)
        self.debug = bool(debug)
        self.current_epoch = 0

        self._validate_class_index(self.free_class_index, 'free_class_index')
        self.ground_layout_indices = self._parse_indices(
            ground_layout_indices, 'ground_layout_indices')
        self.structure_indices = self._parse_indices(
            structure_indices, 'structure_indices')
        self.dynamic_indices = self._parse_indices(
            dynamic_indices, 'dynamic_indices')

        geometry_channels = 8
        blocks = []
        in_channels = geometry_channels
        for _ in range(int(num_convs)):
            blocks.append(
                ConvModule(
                    in_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU', inplace=True)))
            in_channels = hidden_channels
        self.adapter = nn.Sequential(*blocks)
        self.out_conv = nn.Conv2d(
            hidden_channels, self.num_map_classes, kernel_size=1)
        if self.zero_init_output:
            nn.init.constant_(self.out_conv.weight, 0)
            nn.init.constant_(self.out_conv.bias, 0)

    def _validate_class_index(self, index, name):
        if index < 0 or index >= self.num_occ_classes:
            raise ValueError(
                f'{name}={index} is out of range for '
                f'num_occ_classes={self.num_occ_classes}.')

    def _parse_indices(self, indices, name):
        indices = tuple(int(index) for index in indices)
        for index in indices:
            self._validate_class_index(index, name)
        return indices

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def get_beta(self):
        if not self.training:
            return self.beta_o2m_max

        epoch = int(self.current_epoch)
        if epoch <= self.zero_until_epoch:
            return 0.0
        if self.warmup_epochs <= 0:
            return self.beta_o2m_max

        progress = (epoch - self.zero_until_epoch) / float(self.warmup_epochs)
        progress = max(0.0, min(progress, 1.0))
        return self.beta_o2m_max * progress

    def height_pool(self, occ_logits):
        if occ_logits.dim() != 5:
            raise ValueError(
                'OccToMapGeometryAdapter expects occ_logits with shape '
                f'[B, H, W, Z, C], but got {tuple(occ_logits.shape)}.')
        if occ_logits.size(-1) != self.num_occ_classes:
            raise ValueError(
                'OccToMapGeometryAdapter got incompatible occ class count: '
                f'{occ_logits.size(-1)} vs {self.num_occ_classes}.')

        if self.detach_occ:
            occ_logits = occ_logits.detach()

        occ_prob = F.softmax(occ_logits, dim=-1)
        free_prob = occ_prob[..., self.free_class_index]
        occupied_prob = 1.0 - free_prob

        free = free_prob.mean(dim=3)
        occupied = occupied_prob.max(dim=3).values
        ground_layout = self._group_probability(
            occ_prob, self.ground_layout_indices).max(dim=3).values
        structure = self._group_probability(
            occ_prob, self.structure_indices).max(dim=3).values
        dynamic = self._group_probability(
            occ_prob, self.dynamic_indices).max(dim=3).values

        num_z = occupied_prob.size(3)
        z_coord = torch.linspace(
            0.0,
            1.0,
            steps=num_z,
            device=occupied_prob.device,
            dtype=occupied_prob.dtype).view(1, 1, 1, num_z)
        occupied_mass = occupied_prob.sum(dim=3)
        height_mean = (
            occupied_prob * z_coord).sum(dim=3) / (occupied_mass + self.eps)

        valid_height = occupied_prob > self.height_threshold
        height_values = z_coord.expand_as(occupied_prob)
        height_max = torch.where(
            valid_height,
            height_values,
            height_values.new_zeros(())).max(dim=3).values
        occupied_count = occupied_mass

        geometry = torch.stack(
            [
                free,
                occupied,
                ground_layout,
                structure,
                dynamic,
                height_mean,
                height_max,
                occupied_count,
            ],
            dim=1)
        return geometry

    def _group_probability(self, occ_prob, indices):
        if not indices:
            return occ_prob.new_zeros(occ_prob.shape[:-1])
        return occ_prob[..., list(indices)].sum(dim=-1)

    def forward(self, base_map_logits, occ_logits):
        if occ_logits is None:
            raise ValueError(
                'OccToMapGeometryAdapter requires coarse occupancy logits.')

        geometry = self.height_pool(occ_logits)
        if geometry.shape[-2:] != base_map_logits.shape[-2:]:
            geometry = F.interpolate(
                geometry,
                size=base_map_logits.shape[-2:],
                mode='bilinear',
                align_corners=False)

        if self.with_cp and self.training:
            from torch.utils.checkpoint import checkpoint
            hidden = checkpoint(self.adapter, geometry)
        else:
            hidden = self.adapter(geometry)
        delta_map_logits = self.out_conv(hidden)
        beta = float(self.get_beta())
        return base_map_logits + beta * delta_map_logits
