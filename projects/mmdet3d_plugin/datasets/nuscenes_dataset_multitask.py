import torch
from nuscenes.nuscenes import NuScenes

from mmdet3d.datasets import DATASETS

from .nuscenes_dataset_occ import NuScenesDatasetOccpancy


@DATASETS.register_module()
class NuScenesDatasetMultitask(NuScenesDatasetOccpancy):
    """ProtoOcc dataset with occupancy GT plus BEVFusion-style map GT."""

    def __init__(self,
                 map_classes=None,
                 nusc_version='v1.0-trainval',
                 **kwargs):
        super(NuScenesDatasetMultitask, self).__init__(**kwargs)
        self.map_classes = tuple(map_classes) if map_classes is not None else None
        self.scene2location = {}

        if self.map_classes is not None:
            self._build_scene2location(nusc_version)

    def _build_scene2location(self, nusc_version):
        print('[NuScenesDatasetMultitask] Building scene_token -> location cache...')
        nusc = NuScenes(
            version=nusc_version,
            dataroot=self.data_root,
            verbose=False,
        )
        self.scene2location = {}
        for scene in nusc.scene:
            log = nusc.get('log', scene['log_token'])
            self.scene2location[scene['token']] = log['location']
        print(f'[NuScenesDatasetMultitask] Cached {len(self.scene2location)} scene locations.')
        del nusc

    def get_data_info(self, index):
        input_dict = super(NuScenesDatasetMultitask, self).get_data_info(index)

        info = self.data_infos[index]
        location = info.get('location')
        if location is None and self.map_classes is not None:
            scene_token = info.get('scene_token')
            if scene_token is None:
                raise KeyError(
                    'Expected `scene_token` in nuScenes info to recover map location.'
                )
            location = self.scene2location[scene_token]

        if location is not None:
            input_dict['location'] = location

        return input_dict

    def evaluate_map(self, results):
        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            pred = result['masks_bev']
            label = result['gt_masks_bev']

            if not torch.is_tensor(pred):
                pred = torch.as_tensor(pred)
            if not torch.is_tensor(label):
                label = torch.as_tensor(label)

            pred = pred.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-7)

        metrics = {}
        for index, name in enumerate(self.map_classes):
            metrics[f'map/{name}/iou@max'] = ious[index].max().item()
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f'map/{name}/iou@{threshold.item():.2f}'] = iou.item()
        metrics['map/mean/iou@max'] = ious.max(dim=1).values.mean().item()
        return metrics

    def evaluate(self, results, runner=None, show_dir=None, **eval_kwargs):
        if len(results) == 0 or not isinstance(results[0], dict):
            return super(NuScenesDatasetMultitask, self).evaluate(
                results, runner=runner, show_dir=show_dir, **eval_kwargs)

        metrics = eval_kwargs.get('metric', ['miou'])
        if isinstance(metrics, str):
            metrics = [metrics]

        eval_results = {}

        has_occ = 'occ_preds' in results[0]
        has_map = 'masks_bev' in results[0] and 'gt_masks_bev' in results[0]

        if has_occ and any(metric in ['miou', 'ray-iou', 'occ', 'occ-miou'] for metric in metrics):
            occ_metric = 'ray-iou' if 'ray-iou' in metrics else 'miou'
            occ_results = [result['occ_preds'] for result in results]
            eval_results.update(
                super(NuScenesDatasetMultitask, self).evaluate(
                    occ_results,
                    runner=runner,
                    show_dir=show_dir,
                    metric=[occ_metric],
                )
            )

        if has_map and self.map_classes is not None and any(
                metric in ['map', 'map-miou', 'bev-seg', 'bev-seg-miou']
                for metric in metrics):
            eval_results.update(self.evaluate_map(results))

        if eval_results:
            return eval_results

        return super(NuScenesDatasetMultitask, self).evaluate(
            results, runner=runner, show_dir=show_dir, **eval_kwargs)
