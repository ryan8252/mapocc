import torch
import torch.nn.functional as F
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

    def _regrid_map_pred_label(self, pred, label, eval_res, pred_range, gt_range):
        """Re-grid prediction and GT onto a common BEV canvas for a fair,
        same-ruler comparison against papers that evaluate at a different
        range/resolution (e.g. MAESTRO uses the BEVFusion [-50,50]@0.5m grid
        while this model predicts [-40,40]@0.4m).

        - `pred` is resampled from its native grid to a `pred_range`@`eval_res`
          canvas (area-average over probabilities), then zero-padded out to the
          `gt_range`@`eval_res` canvas. The padded ring is treated as background
          because the model has no prediction beyond `pred_range`.
        - `label` is resampled to the `gt_range`@`eval_res` canvas with an
          occupied-if-any rule, matching how a thin structure rasterizes onto a
          coarser cell.
        """
        gt_cells = int(round(2.0 * gt_range / eval_res))
        pred_cells = int(round(2.0 * pred_range / eval_res))

        pred = pred[None].float()
        pred = F.interpolate(pred, size=(pred_cells, pred_cells), mode='area')[0]
        if gt_cells > pred_cells:
            pad = gt_cells - pred_cells
            left = pad // 2
            pred = F.pad(pred, (left, pad - left, left, pad - left), value=0.0)

        if label.shape[-1] != gt_cells or label.shape[-2] != gt_cells:
            label = F.interpolate(
                label[None].float(), size=(gt_cells, gt_cells), mode='area')[0]
            label = label > 0
        return pred, label.bool()

    def evaluate_map(self, results, eval_res=None, pred_range=40.0,
                     gt_range=None):
        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)
        if eval_res is not None and gt_range is None:
            gt_range = pred_range

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

            pred = pred.detach().float()
            label = label.detach()

            if eval_res is not None:
                pred, label = self._regrid_map_pred_label(
                    pred, label, eval_res, pred_range, gt_range)

            pred = pred.reshape(num_classes, -1)
            label = label.bool().reshape(num_classes, -1)

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
            # Native-grid map metrics (whatever grid the GT pipeline produced).
            eval_results.update(self.evaluate_map(results))

            # Optional same-ruler re-grid(s) for fair comparison against papers
            # evaluated at a different range/resolution (e.g. MAESTRO's
            # BEVFusion [-50,50]@0.5m grid). Pass via --eval-options:
            #   map_eval_res=0.5 map_pred_range=40 map_gt_range=40   (Test R)
            #   map_eval_res=0.5 map_pred_range=40 map_gt_range=50   (Test M)
            map_eval_res = eval_kwargs.get('map_eval_res', None)
            if map_eval_res is not None:
                pred_range = float(eval_kwargs.get('map_pred_range', 40.0))
                gt_range = eval_kwargs.get('map_gt_range', None)
                gt_range = float(gt_range) if gt_range is not None else pred_range
                res_list = (map_eval_res if isinstance(map_eval_res, (list, tuple))
                            else [map_eval_res])
                for res in res_list:
                    res = float(res)
                    grid_metrics = self.evaluate_map(
                        results, eval_res=res, pred_range=pred_range,
                        gt_range=gt_range)
                    tag = f'map@{res:g}m_pr{pred_range:g}_gr{gt_range:g}/'
                    eval_results.update({
                        k.replace('map/', tag, 1): v
                        for k, v in grid_metrics.items()})

        if eval_results:
            return eval_results

        return super(NuScenesDatasetMultitask, self).evaluate(
            results, runner=runner, show_dir=show_dir, **eval_kwargs)
