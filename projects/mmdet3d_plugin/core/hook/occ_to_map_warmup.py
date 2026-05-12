from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class OccToMapWarmupHook(Hook):
    """Set the 1-based epoch for the Occ-to-Map residual schedule."""

    def _unwrap_model(self, model):
        while hasattr(model, 'module') and model.module is not model:
            model = model.module
        return model

    def _set_epoch(self, runner, epoch):
        model = self._unwrap_model(runner.model)
        if hasattr(model, 'set_o2m_epoch'):
            model.set_o2m_epoch(epoch)
            beta = None
            if getattr(model, 'occ_to_map_adapter', None) is not None:
                beta = model.occ_to_map_adapter.get_beta()
            if beta is not None:
                runner.logger.info(
                    f'OccToMapWarmupHook set o2m epoch={epoch}, beta={beta:.4f}')

    def before_run(self, runner):
        self._set_epoch(runner, 0)

    def before_train_epoch(self, runner):
        self._set_epoch(runner, runner.epoch + 1)
