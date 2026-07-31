from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class CFVProtoTSFGWarmupHook(Hook):
    """Runtime gamma scheduler for CFVPrototypeTSFGMapFusion."""

    def __init__(self,
                 start_epoch=0,
                 end_epoch=3,
                 start_value=0.0,
                 end_value=1.0):
        super().__init__()
        if end_epoch < start_epoch:
            raise ValueError('end_epoch must be >= start_epoch.')
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_value = float(start_value)
        self.end_value = float(end_value)

    def _unwrap_model(self, model):
        while hasattr(model, 'module'):
            model = model.module
        return model

    def _value_at_epoch(self, epoch):
        if epoch <= self.start_epoch:
            return self.start_value
        if epoch >= self.end_epoch:
            return self.end_value
        span = max(self.end_epoch - self.start_epoch, 1)
        ratio = float(epoch - self.start_epoch) / float(span)
        return self.start_value + ratio * (self.end_value - self.start_value)

    def _set_gamma(self, runner):
        model = self._unwrap_model(runner.model)
        value = self._value_at_epoch(runner.epoch)
        if not hasattr(model, 'set_cfv_proto_tsfg_gamma'):
            runner.logger.warning(
                'CFVProtoTSFGWarmupHook found no '
                'set_cfv_proto_tsfg_gamma() on model %s.',
                model.__class__.__name__)
            return
        updated = model.set_cfv_proto_tsfg_gamma(value)
        if updated:
            runner.logger.info(
                'CFVProtoTSFGWarmupHook set gamma=%.4f at epoch %s.',
                value, runner.epoch)

    def before_run(self, runner):
        self._set_gamma(runner)

    def before_train_epoch(self, runner):
        self._set_gamma(runner)
