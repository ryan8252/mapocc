# ProtoOcc Train Eval Record Runbook

This runbook documents the exact local workflow used on 2026-06-01 for a
1-epoch ProtoOcc smoke train, distributed eval, and `result.md` recording.
Future automation should read this file first, then substitute the requested
config, work directory, epoch count, dataset split, and output path.

## Purpose

Use this when the user asks Codex to train a ProtoOcc config, evaluate the
resulting checkpoint, and write a reproducible `result.md`.

The default policy is:

- Train in the requested `WORK_DIR`.
- Evaluate the newly produced EMA checkpoint if it exists.
- Write `result.md` into the same `WORK_DIR` unless the user explicitly gives a
  different `RESULT_DIR`.
- If `RESULT_DIR` differs from `WORK_DIR`, the result file must clearly state
  the actual checkpoint source.

## Environment

Repository:

```bash
cd /media/robot/16TB/ARTC2026/Ryan/ProtoOcc
```

Conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mapocc
```

Codex note:

- `tools/dist_train.sh` and `tools/dist_test.sh` use `torch.distributed`.
- In the sandbox, local rendezvous sockets can fail with `Operation not
  permitted`.
- If that happens, rerun the same command with escalated permissions.

## Inputs To Substitute

```bash
CONFIG=projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_bevfusion_loss_focal.py
WORK_DIR=work_dirs/smoke_map_neck_bevfusion_loss_focal_weight128_1quarter_4090
RESULT_DIR=$WORK_DIR
GPUS=1
TRAIN_PORT=29511
EVAL_PORT=29512
EPOCHS=1
TRAIN_ANN=data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl
SAMPLES_PER_GPU=2
EVAL_METRICS="miou map-miou"
```

For full training, replace `TRAIN_ANN`, `SAMPLES_PER_GPU`, `EPOCHS`, and the
work directory deliberately. Do not reuse a smoke-run work directory for a
paper-result run.

## Preflight

Check that the config exists and inspect any existing checkpoint state:

```bash
test -f "$CONFIG"
mkdir -p "$WORK_DIR"
find "$WORK_DIR" -maxdepth 1 -type f \( -name 'epoch_*.pth' -o -name 'latest.pth' -o -name '*.log' \) -printf '%p\n'
```

If `WORK_DIR` already contains checkpoints, decide whether this is a fresh run
or a continuation:

- Fresh run: use a new work directory or accept that the directory will contain
  multiple logs/checkpoints.
- Strict continuation: use `--resume-from latest.pth` or a non-EMA checkpoint
  such as `epoch_2.pth`, and set `runner.max_epochs` above the completed epoch.
- Warm start only: use `load_from` in config or cfg-options if supported by the
  local config path. Do not call EMA-only checkpoints a strict resume.

## Train

Template:

```bash
PORT=$TRAIN_PORT bash tools/dist_train.sh \
  "$CONFIG" \
  "$GPUS" \
  --work-dir "$WORK_DIR" \
  --cfg-options \
  data.train.ann_file="$TRAIN_ANN" \
  data.samples_per_gpu="$SAMPLES_PER_GPU" \
  runner.max_epochs="$EPOCHS" \
  evaluation.interval=999 \
  checkpoint_config.interval=1
```

Command used on 2026-06-01:

```bash
PORT=29511 bash tools/dist_train.sh \
  projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_bevfusion_loss_focal.py \
  1 \
  --work-dir work_dirs/smoke_map_neck_bevfusion_loss_focal_weight128_1quarter_4090 \
  --cfg-options \
  data.train.ann_file=data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl \
  data.samples_per_gpu=2 \
  runner.max_epochs=1 \
  evaluation.interval=999 \
  checkpoint_config.interval=1
```

Successful train evidence from this run:

- Log: `work_dirs/smoke_map_neck_bevfusion_loss_focal_weight128_1quarter_4090/20260601_221922.log`
- Checkpoint: `work_dirs/smoke_map_neck_bevfusion_loss_focal_weight128_1quarter_4090/epoch_1.pth`
- EMA checkpoint: `work_dirs/smoke_map_neck_bevfusion_loss_focal_weight128_1quarter_4090/epoch_1_ema.pth`
- `latest.pth` points to the non-EMA checkpoint.

## Select Checkpoint For Eval

Preferred:

```bash
CKPT="$WORK_DIR/epoch_${EPOCHS}_ema.pth"
test -f "$CKPT"
```

Fallbacks:

- If EMA does not exist, use `"$WORK_DIR/epoch_${EPOCHS}.pth"`.
- If evaluating a resumed or multi-epoch run, use the user-requested checkpoint
  exactly.
- Always record the exact checkpoint path in `result.md`.

## Eval

Template:

```bash
PORT=$EVAL_PORT bash tools/dist_test.sh \
  "$CONFIG" \
  "$CKPT" \
  "$GPUS" \
  --eval $EVAL_METRICS
```

Command used on 2026-06-01:

```bash
PORT=29512 bash tools/dist_test.sh \
  ./projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_bevfusion_loss_focal.py \
  ./work_dirs/smoke_map_neck_bevfusion_loss_focal_weight128_1quarter_4090/epoch_1_ema.pth \
  1 \
  --eval miou map-miou
```

Expected behavior:

- First stage runs inference over the val set, e.g. `6019/6019`.
- Then `metric = miou` starts CPU-side metric aggregation.
- The command can look quiet between inference and metric output. Check process
  state before assuming it is stuck.

## Parse Eval Output

Extract these fields from stdout:

- OCC summary: line matching `===> mIoU of`.
- OCC per-class lines: lines matching `===> <class> - IoU = <value>`.
- Map summary: dictionary keys like `map/mean/iou@max`.
- Map class metrics: dictionary keys like `map/drivable_area/iou@max`.

This run produced:

```text
===> mIoU of 6019 samples: 27.28
map/mean/iou@max: 0.17407339811325073
map/drivable_area/iou@max: 0.6107810735702515
map/ped_crossing/iou@max: 0.0
map/walkway/iou@max: 0.26053157448768616
map/stop_line/iou@max: 9.741353278513998e-05
map/carpark_area/iou@max: 0.04444609209895134
map/divider/iou@max: 0.12858428061008453
```

## Write result.md

Create or update:

```bash
RESULT_MD="$RESULT_DIR/result.md"
mkdir -p "$RESULT_DIR"
```

If the file does not exist, create it. If it exists, append a new section with
the date, config, checkpoint, commands, and metrics. Never overwrite earlier
results unless the user explicitly asks.

Minimum required sections:

````markdown
# Eval Result - <run name>

Date: <YYYY-MM-DD>

## Run

- Config: `<CONFIG>`
- Work dir: `<WORK_DIR>`
- Train log: `<latest train log>`
- Checkpoint: `<non-EMA checkpoint if present>`
- EMA checkpoint: `<EMA checkpoint if used>`
- Eval checkpoint: `<CKPT>`

## Commands

```bash
<train command>
```

```bash
<eval command>
```

## Summary

- OCC mIoU: `<value>`
- Map mean iou@max: `<value>`
- Best map classes: `<class=value, ...>`
- Thin classes: `ped_crossing=<value>`, `stop_line=<value>`, `divider=<value>`

## OCC IoU

| Class | IoU |
| --- | ---: |
| ... | ... |

## Map IoU

| Class | iou@max | iou@0.35 | iou@0.40 | iou@0.45 | iou@0.50 | iou@0.55 | iou@0.60 | iou@0.65 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ... | ... |
````

Actual result file for this run:

```text
work_dirs/smoke_map_neck_bevfusion_loss_focal_weight128_1quarter_4090/result.md
```

## Completion Report

When finished, report these items to the user:

- Whether train exited with code `0`.
- Whether eval exited with code `0`.
- Exact checkpoint evaluated.
- Exact `result.md` path.
- Key metrics: OCC mIoU, map mean iou@max, and thin-class map IoUs.
- Any path mismatch, missing checkpoint, or sandbox escalation used.

Example completion summary from this run:

```text
Train completed for 1 epoch.
Eval completed on epoch_1_ema.pth.
OCC mIoU: 27.28.
Map mean iou@max: 0.174073.
Thin classes: ped_crossing=0.0, stop_line=0.000097, divider=0.128584.
Result file: work_dirs/smoke_map_neck_bevfusion_loss_focal_weight128_1quarter_4090/result.md.
```

## Future Prompt Shape

The user can ask:

```text
Read train_eval_record_runbook.md, then run this ProtoOcc config:
CONFIG=<config path>
WORK_DIR=<work dir>
EPOCHS=<n>
TRAIN_ANN=<ann pkl or default>
SAMPLES_PER_GPU=<n>
Evaluate epoch_<n>_ema.pth with miou map-miou and update result.md.
```

Codex should then execute the workflow end to end, using the substitutions from
the prompt and this runbook.
