# PGBR Optional Submodule Convention

日期：2026-04-23

## 背景

PGBR (Prototype-Grounded BEV Refinement) 的 ablation 結果目前不利：

| 版本 | Occ mIoU | Map mIoU |
| --- | ---: | ---: |
| With PGBR | 36.96 | 31.46 |
| Canonical ProtoMapHead, no PGBR | 37.61 | 32.95 |

所以現在的約定是：`ProtoMapHead` 本體只負責 prototype query / decoder。PGBR 降級成 optional submodule，只在 config 明確提供 `pgbr_cfg` 時才會建立。

## Config 約定

canonical config 不寫任何 PGBR 相關欄位：

```python
proto_map_head=dict(
    type='ProtoMapHead',
    in_channels=voxel_out_channels,
    hidden_channels=voxel_out_channels * 2,
    # no pgbr_cfg here
)
```

需要做 PGBR ablation 時才 opt-in：

```python
proto_map_head=dict(
    pgbr_cfg=dict(
        temperature=1.0,
        detach_query=True,
    )
)
```

不要再在 canonical config 裡寫 `use_bev_refinement=False`。不傳 `pgbr_cfg` 就是不開，這樣架構意義比較乾淨，也避免未來新增其他 refiner 時所有 config 都要塞一個 false flag。

## 目前檔案角色

| 檔案 | 角色 |
| --- | --- |
| `ProtoOcc_proto_map_head.py` | canonical ProtoMapHead，預設不建立 PGBR |
| `ProtoOcc_proto_map_head_pgbr.py` | PGBR ablation，只覆寫 `pgbr_cfg` |
| `ProtoOcc_proto_map_head_map_neck.py` | map-specific BEV neck 主實驗，繼承 canonical no-PGBR |
| `ProtoOcc_proto_map_head_map_neck_pgbr.py` | map-neck + PGBR ablation，只覆寫 `pgbr_cfg` |
| `ProtoOcc_proto_map_head_map_neck_256.py` | map-neck channel ablation，仍預設 no-PGBR |
| `ProtoOcc_proto_map_head_map_neck_detach.py` | map-neck detach ablation，仍預設 no-PGBR |

## Model 實作約定

- `PrototypeGroundedBEVRefiner` 是獨立 submodule。
- `ProtoMapHead.pgbr_refiner` 預設為 `None`。
- 只有 `pgbr_cfg` 存在時，`ProtoMapHead` 才 instantiate `PrototypeGroundedBEVRefiner`。
- forward 端透過 `_apply_pgbr()` 包一層；沒有 refiner 時直接回傳原本 `mask_feat`。
- 舊 config 的 `use_bev_refinement=True` 仍保留相容，但只作為 legacy path；新 config 不使用它。
- 舊 checkpoint 裡 inline PGBR 權重會在 load 時對應到 `pgbr_refiner.*`，canonical no-PGBR model 會忽略這些舊權重。

## 繼承關係

```text
ProtoOcc_proto_map_head.py                    canonical, no pgbr_cfg
│
├── ProtoOcc_proto_map_head_pgbr.py           PGBR ablation
│
└── ProtoOcc_proto_map_head_map_neck.py       map-specific neck, no pgbr_cfg
    ├── ProtoOcc_proto_map_head_map_neck_pgbr.py     PGBR ablation
    ├── ProtoOcc_proto_map_head_map_neck_256.py      channel ablation
    └── ProtoOcc_proto_map_head_map_neck_detach.py   detach ablation
```
