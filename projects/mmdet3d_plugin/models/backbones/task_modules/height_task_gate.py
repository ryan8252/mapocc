import torch
from torch import nn

from mmdet3d.models import BACKBONES


@BACKBONES.register_module()
class HeightTaskGate(nn.Module):
    """Task-specific residual gate over LSS height bins.

    The zero-initialized final conv keeps step 0 identical to the input while
    the fixed non-zero gate scale still lets the gate branch receive gradients.
    """

    def __init__(self,
                 in_channels=80,
                 reduction=4,
                 gate_scale=0.1,
                 zero_init_last=True):
        super().__init__()
        hidden_channels = max(in_channels // reduction, 1)
        self.gate_scale = gate_scale

        self.gate = nn.Sequential(
            nn.Conv3d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, 2, kernel_size=1))

        if zero_init_last:
            nn.init.constant_(self.gate[-1].weight, 0)
            nn.init.constant_(self.gate[-1].bias, 0)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(
                'HeightTaskGate expects input shape [B, C, Z, H, W], '
                f'but got {tuple(x.shape)}.')

        gates = torch.tanh(self.gate(x))
        g_occ = gates[:, 0:1]
        g_map = gates[:, 1:2]

        x_occ = x * (1.0 + self.gate_scale * g_occ)
        x_map = x * (1.0 + self.gate_scale * g_map)
        return x_occ, x_map
