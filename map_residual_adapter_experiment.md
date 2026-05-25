# Per-scale map residual adapter experiment

## Scope

Experiment target:

- CNN BEVSegHead
- 128-channel map-specific BEV neck
- `map_loss_weight=4.0`
- Per-scale zero-init map residual adapter before the map neck

This change does not modify ProtoMapHead, VAMI, datasets, map classes, loss
definitions, EMA hook, optimizer, evaluation, or training scripts.

## Files

Modified:

- `projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py`

New:

- `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_adapter.py`
- `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_adapter_detach.py`
- `map_residual_adapter_experiment.md`

## Architecture insertion point

The adapter is inserted only on the map path in `Dual_Branch_Encoder.forward()`:

```text
pooled_x
  -> down_sample_for_3d_pooling
  -> bev_encoder_backbone
  -> multi_scale_bev

occupancy path:
  multi_scale_bev
  -> existing HFM / bev_encoder_neck / voxelize_module
  -> occupancy heads

map path:
  multi_scale_bev
  -> optional detach_map_feature
  -> PerScaleMapResidualAdapter
  -> existing map_bev_encoder_neck
  -> BEVSegHead
```

When `map_residual_adapter=None`, the map path remains:

```text
multi_scale_bev -> optional detach_map_feature -> map_bev_encoder_neck
```

## Adapter details

`PerScaleMapResidualAdapter` is registered as a backbone and accepts
three BEV feature scales with channels `[160, 320, 640]`.

Each scale uses:

```text
1x1 Conv2d(C, C)
BatchNorm2d
ReLU
depthwise 3x3 Conv2d(C, C, groups=C, padding=1)
BatchNorm2d
ReLU
1x1 Conv2d(C, C)
```

The final `1x1 Conv2d` in every block is zero-initialized, so the initial
adapter output is identical to the input:

```text
output = feat + residual_scale * 0 = feat
```

For `detach_input=True`, the adapter uses:

```text
source = feat.detach()
output = source + residual_scale * adapter(source)
```

Checkpointing inside the adapter is skipped automatically when the selected
source feature does not require gradients, so the detach variant still allows
adapter parameters to learn.

## Configs

Main experiment:

- `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_adapter.py`
- Inherits `./ProtoOcc_multi_cnn_head_map_neck.py`
- Sets `map_loss_weight=4.0`
- Adds `map_residual_adapter` with `detach_input=False`

Safer detach variant:

- `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_adapter_detach.py`
- Inherits the main adapter config
- Sets only `detach_input=True`

Both configs preserve:

- CNN map head
- `map_bev_encoder_neck.out_channels=128`
- `BEVSegHead.in_channels=128`
- `BEVSegHead.hidden_channels=128`
- Existing occupancy branch config

## Validation

Static checks:

```text
python -m py_compile projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py
git diff --check
```

Both passed.

Config loading:

```text
ProtoOcc_multi_cnn_head_map_neck.py
  map_loss_weight = 1.0
  bev_seg_head.in_channels = 128
  bev_seg_head.hidden_channels = 128
  map neck out_channels = 128
  adapter = None

ProtoOcc_multi_cnn_head_map_neck_adapter.py
  map_loss_weight = 4.0
  bev_seg_head.in_channels = 128
  bev_seg_head.hidden_channels = 128
  map neck out_channels = 128
  adapter.detach_input = False

ProtoOcc_multi_cnn_head_map_neck_adapter_detach.py
  map_loss_weight = 4.0
  bev_seg_head.in_channels = 128
  bev_seg_head.hidden_channels = 128
  map neck out_channels = 128
  adapter.detach_input = True
```

Encoder build:

```text
ProtoOcc_multi_cnn_head_map_neck.py
  encoder = Dual_Branch_Encoder
  map_residual_adapter = None

ProtoOcc_multi_cnn_head_map_neck_adapter.py
  encoder = Dual_Branch_Encoder
  map_residual_adapter = PerScaleMapResidualAdapter
  detach_input = False

ProtoOcc_multi_cnn_head_map_neck_adapter_detach.py
  encoder = Dual_Branch_Encoder
  map_residual_adapter = PerScaleMapResidualAdapter
  detach_input = True
```

Adapter unit check:

```text
input shapes =
  [(2, 160, 100, 100), (2, 320, 50, 50), (2, 640, 25, 25)]

adapter output shapes =
  [(2, 160, 100, 100), (2, 320, 50, 50), (2, 640, 25, 25)]

max init diff = 0.0
last conv zero = True
adapter params = 1093120
detach output requires_grad = [True, True, True]
```

Detector build:

- Direct full detector build failed on this machine because the existing
  `Prototype_Query_Decoder_nuScenes` constructor calls `.cuda()` and no CUDA
  GPU is visible.
- With a temporary CPU-only `torch.Tensor.cuda` no-op monkeypatch, the baseline
  and adapter configs built successfully:

```text
ProtoOcc_multi_cnn_head_map_neck.py
  model = ProtoOccCnnSegHead
  adapter = None
  map_loss_weight = 1.0

ProtoOcc_multi_cnn_head_map_neck_adapter.py
  model = ProtoOccCnnSegHead
  adapter = PerScaleMapResidualAdapter
  map_loss_weight = 4.0

ProtoOcc_multi_cnn_head_map_neck_adapter_detach.py
  model = ProtoOccCnnSegHead
  adapter = PerScaleMapResidualAdapter
  detach_input = True
  map_loss_weight = 4.0
```

## Old behavior

The old map-neck config still has no `map_residual_adapter` key after config
loading, and `Dual_Branch_Encoder.map_residual_adapter` builds as `None`.
Therefore the old config keeps the same branch logic and does not route through
the adapter.

## Parameter count

The adapter adds `1,093,120` trainable parameters with channels
`[160, 320, 640]`.

## Risks

- The no-detach adapter lets map loss gradients pass through the adapter into
  the shared BEV backbone, so occupancy may move. Use the detach config if this
  hurts occupancy.
- The zero-init residual protects initialization equivalence, but training can
  still learn map-specific changes that interact with the shared BEV features.
- Full detector construction on CPU still depends on the temporary `.cuda()`
  monkeypatch because of an existing PQD CUDA allocation in the repo.
