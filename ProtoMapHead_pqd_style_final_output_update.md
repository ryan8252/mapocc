# ProtoMapHead PQD-style Final Output Update

Date: 2026-05-15

## Motivation

The previous `ProtoMapHead` used six class-fixed queries directly:

```text
query_k -> mask_embed_k -> final_masks[:, k]
```

This kept the map output simple, but it did not mirror ProtoOcc PQD's final
decoding pattern where each scene-aware query also predicts a class-confidence
branch. The new version keeps the map queries class-specific, but adds a
PQD-inspired correction/confidence branch before producing the final map logits.

## What Changed

File changed:

```text
projects/mmdet3d_plugin/models/dense_heads/proto_map_head.py
```

### 1. Added `cls_embed`

`ProtoMapHead` now has:

```text
cls_embed: query [B, K, C] -> class logits [B, K, K]
```

For BEV map segmentation:

```text
K = 6 map classes
C = hidden_channels, usually 128 for the 128ch map-neck config
```

The branch is initialized with zero weights and zero bias. A fixed identity
bias is added during final decoding so the model starts close to the old
class-fixed behavior:

```text
query 0 -> class 0
query 1 -> class 1
...
query 5 -> class 5
```

### 2. Changed final mask composition

Old final output:

```text
mask_embed = mask_embed(query)                  # [B, K, C]
final_masks = einsum(mask_embed, mask_feature)  # [B, K, H, W]
```

New final output:

```text
cls_logits  = cls_embed(query)                  # [B, K, K]
class_prob  = softmax(cls_logits + identity_bias, dim=-1)
mask_embed  = mask_embed(query)                 # [B, K, C]
query_masks = einsum(mask_embed, mask_feature)  # [B, K, H, W]
final_masks = einsum(class_prob, query_masks)   # [B, K, H, W]
```

This is the map-side analogue of PQD's class-probability times spatial-mask
composition, but it remains a multi-label map prediction.

### 3. Final map output is still sigmoid multi-label

No map loss or inference API was changed:

```text
final_masks [B, 6, H, W] -> sigmoid(final_masks)
```

The map head still trains with per-class BCE/Focal/Dice-style losses. It does
not use occupancy-style argmax, and it does not add a no-object class.

## Tensor Path

For the 128-channel map-neck setting:

```text
map_feature       [B, 128, 200, 200]
mask_feature      [B, 128, 200, 200]
mask_feat         [B, 32, 200, 200]
coarse_pred       [B, 6, 200, 200]
for_query         [6, 32]
ema_query         [6, 32]
query_feat        [6, 128]
self-attn query   [6, B, 128]
cls_logits        [B, 6, 6]
class_prob        [B, 6, 6]
query_masks       [B, 6, 200, 200]
final_masks       [B, 6, 200, 200]
map_probs         [B, 6, 200, 200]
```

## Design Boundary

This is not a full conversion to occupancy PQD:

- Queries remain class-specific.
- `cls_embed` is used as class correction/confidence, not as a single-label
  semantic classifier for final argmax.
- Final BEV map output remains six sigmoid multi-label channels.
- No RPL, no no-object query class, and no map argmax are added.

This should be treated as an ablation against the previous class-fixed
`ProtoMapHead` final decoder.
