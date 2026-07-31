# Copyright (c) OpenMMLab. All rights reserved.
from .ema import MEGVIIEMAHook
from .utils import is_parallel
from .sequentialcontrol import SequentialControlHook
from .syncbncontrol import SyncbnControlHook
from .cfv_proto_tsfg_warmup import CFVProtoTSFGWarmupHook
from .map_training_schedule import (
    MapFirstStageLossHook, MapLossWeightScheduleHook)

__all__ = ['MEGVIIEMAHook', 'SequentialControlHook', 'is_parallel',
           'SyncbnControlHook', 'CFVProtoTSFGWarmupHook',
           'MapFirstStageLossHook', 'MapLossWeightScheduleHook']
