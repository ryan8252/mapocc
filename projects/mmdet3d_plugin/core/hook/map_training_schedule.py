from mmcv.runner.hooks import HOOKS, Hook


def _unwrap_model(model):
    while hasattr(model, 'module'):
        model = model.module
    return model


def _resolve_attr(root, path):
    obj = root
    parts = path.split('.')
    for part in parts[:-1]:
        if not hasattr(obj, part):
            return None, None
        obj = getattr(obj, part)
    if not hasattr(obj, parts[-1]):
        return None, None
    return obj, parts[-1]


@HOOKS.register_module()
class MapLossWeightScheduleHook(Hook):
    """Linearly schedule ``model.map_loss_weight`` by epoch."""

    def __init__(self,
                 start_epoch=0,
                 end_epoch=12,
                 start_value=4.0,
                 end_value=8.0):
        super().__init__()
        if end_epoch < start_epoch:
            raise ValueError('end_epoch must be >= start_epoch.')
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self._last_value = None

    def _value_at_epoch(self, epoch):
        if epoch <= self.start_epoch:
            return self.start_value
        if epoch >= self.end_epoch:
            return self.end_value
        span = max(self.end_epoch - self.start_epoch, 1)
        ratio = float(epoch - self.start_epoch) / float(span)
        return self.start_value + ratio * (self.end_value - self.start_value)

    def _apply(self, runner):
        model = _unwrap_model(runner.model)
        if not hasattr(model, 'map_loss_weight'):
            runner.logger.warning(
                'MapLossWeightScheduleHook found no map_loss_weight on %s.',
                model.__class__.__name__)
            return

        value = self._value_at_epoch(runner.epoch)
        model.map_loss_weight = value
        if self._last_value is None or abs(value - self._last_value) > 1e-6:
            runner.logger.info(
                'MapLossWeightScheduleHook set map_loss_weight=%.4f '
                'at epoch %s.', value, runner.epoch)
            self._last_value = value

    def before_run(self, runner):
        self._apply(runner)

    def before_train_epoch(self, runner):
        self._apply(runner)


@HOOKS.register_module()
class MapFirstStageLossHook(Hook):
    """Disable OCC-side losses for the first stage, then restore them.

    The forward path is unchanged. This only changes scalar loss weights, so
    the same checkpoint can be evaluated with the normal joint architecture.
    """

    DEFAULT_OCC_TARGETS = (
        'cnn3d_decoder.loss_weight',
        'prototype_query_decoder.loss_cls.loss_weight',
        'prototype_query_decoder.loss_mask.loss_weight',
        'prototype_query_decoder.loss_dice.loss_weight',
    )

    DEFAULT_DEPTH_TARGETS = (
        'depth_net.loss_depth_weight',
    )

    def __init__(self,
                 map_first_epochs=12,
                 map_loss_weight=4.0,
                 disable_occ_loss=True,
                 disable_depth_loss=False,
                 extra_zero_targets=None):
        super().__init__()
        if map_first_epochs < 0:
            raise ValueError('map_first_epochs must be >= 0.')
        self.map_first_epochs = int(map_first_epochs)
        self.map_loss_weight = float(map_loss_weight)
        self.disable_occ_loss = bool(disable_occ_loss)
        self.disable_depth_loss = bool(disable_depth_loss)
        self.extra_zero_targets = tuple(extra_zero_targets or ())
        self._original_values = {}
        self._missing_targets = set()
        self._stage = None

    def _target_paths(self):
        paths = []
        if self.disable_occ_loss:
            paths.extend(self.DEFAULT_OCC_TARGETS)
        if self.disable_depth_loss:
            paths.extend(self.DEFAULT_DEPTH_TARGETS)
        paths.extend(self.extra_zero_targets)
        return tuple(paths)

    def _store_originals(self, model):
        for path in self._target_paths():
            owner, name = _resolve_attr(model, path)
            if owner is None:
                self._missing_targets.add(path)
                continue
            if path not in self._original_values:
                self._original_values[path] = getattr(owner, name)

    def _set_targets(self, model, active):
        for path in self._target_paths():
            owner, name = _resolve_attr(model, path)
            if owner is None:
                self._missing_targets.add(path)
                continue
            value = 0.0 if active else self._original_values.get(path)
            if value is not None:
                setattr(owner, name, value)

    def _apply(self, runner):
        model = _unwrap_model(runner.model)
        self._store_originals(model)

        if hasattr(model, 'map_loss_weight'):
            model.map_loss_weight = self.map_loss_weight
        else:
            runner.logger.warning(
                'MapFirstStageLossHook found no map_loss_weight on %s.',
                model.__class__.__name__)

        active = runner.epoch < self.map_first_epochs
        self._set_targets(model, active)
        stage = 'map_first' if active else 'joint'
        if stage != self._stage:
            runner.logger.info(
                'MapFirstStageLossHook switched to %s stage at epoch %s '
                '(map_loss_weight=%.4f).',
                stage, runner.epoch, self.map_loss_weight)
            self._stage = stage

        if self._missing_targets:
            runner.logger.warning(
                'MapFirstStageLossHook skipped missing targets: %s',
                ', '.join(sorted(self._missing_targets)))
            self._missing_targets.clear()

    def before_run(self, runner):
        self._apply(runner)

    def before_train_epoch(self, runner):
        self._apply(runner)
