import os

import mmcv
import numpy as np
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

    def save_map_results(self, results, output_dir):
        """Save compact map-only prediction files indexed by scene and token."""
        if len(results) != len(self.data_infos):
            raise ValueError(
                'Cannot align map results with nuScenes metadata: '
                f'got {len(results)} results for {len(self.data_infos)} samples.')

        map_loader = next(
            (transform for transform in getattr(self.pipeline, 'transforms', ())
             if hasattr(transform, 'xbound') and hasattr(transform, 'ybound')),
            None,
        )
        if map_loader is None:
            raise RuntimeError(
                'Cannot export map geometry: LoadBEVSegmentation was not '
                'found in the dataset pipeline.')

        classes = np.asarray(self.map_classes)
        xbound = np.asarray(map_loader.xbound, dtype=np.float32)
        ybound = np.asarray(map_loader.ybound, dtype=np.float32)
        for index, result in enumerate(results):
            if 'masks_bev' not in result or 'gt_masks_bev' not in result:
                raise KeyError(
                    f'Map result {index} is missing masks_bev/gt_masks_bev.')

            pred = result['masks_bev']
            label = result['gt_masks_bev']
            if torch.is_tensor(pred):
                pred = pred.detach().cpu().numpy()
            if torch.is_tensor(label):
                label = label.detach().cpu().numpy()

            info = self.data_infos[index]
            sample_token = str(info['token'])
            scene_name = info.get('scene_name')
            if scene_name is None:
                occ_path = str(info.get('occ_path', ''))
                scene_parts = [part for part in occ_path.split('/')
                               if part.startswith('scene-')]
                scene_name = (scene_parts[0] if scene_parts
                              else info.get('scene_token', 'unknown-scene'))
            scene_name = str(scene_name)

            sample_dir = os.path.join(output_dir, scene_name, sample_token)
            mmcv.mkdir_or_exist(sample_dir)
            np.savez_compressed(
                os.path.join(sample_dir, 'map.npz'),
                probs=np.asarray(pred, dtype=np.float32),
                gt=np.asarray(label, dtype=np.uint8),
                classes=classes,
                xbound=xbound,
                ybound=ybound,
                sample_token=np.asarray(sample_token),
                scene_name=np.asarray(scene_name),
                axis_convention=np.asarray('protoocc'),
            )

        print(f'Saved {len(results)} BEV map predictions to {output_dir}')

    @staticmethod
    def _build_ring_masks(h, w, map_range, ring_edges):
        """Boolean (h*w,) masks per radial distance ring.

        The BEV grid is assumed to cover [-map_range, map_range] in both axes
        with `h`/`w` cells. Cell centres are used to compute the ego-centric
        radial distance; rings are [edge_i, edge_{i+1}). Radial rings are
        rotation-invariant, so axis order / flips do not matter here.
        """
        cy = (torch.arange(h).float() + 0.5) / h * (2.0 * map_range) - map_range
        cx = (torch.arange(w).float() + 0.5) / w * (2.0 * map_range) - map_range
        yy, xx = torch.meshgrid(cy, cx, indexing='ij')
        dist = torch.sqrt(xx ** 2 + yy ** 2).reshape(-1)
        masks = []
        for lo, hi in zip(ring_edges[:-1], ring_edges[1:]):
            masks.append((dist >= lo) & (dist < hi))
        return masks

    def evaluate_map_rings(self, results, map_range=50.0,
                           ring_edges=(0.0, 20.0, 40.0, 50.0, 71.0),
                           tolerances=(1, 2)):
        """Per-distance-ring map IoU on the native eval grid.

        Diagnoses where the map gap lives (e.g. near-field vs the 40-50m ring
        that is camera-starved when LSS lift/depth stop at 45m). Same TP/FP/FN
        IoU definition as `evaluate_map`, just restricted to each radial ring.

        Besides per-ring IoU it reports, at each ring's own best threshold,
        strict precision/recall plus alignment-tolerant variants: for each
        tolerance `d` (eval-grid pixels; 1 px = 0.5 m on the native
        [-50, 50] @ 0.5 m grid), `prec_tol` counts a predicted cell as correct
        if it lies within d px of any GT cell (GT dilated), `rec_tol` counts a
        GT cell as covered if a prediction lies within d px (pred dilated).
        A large strict->tolerant precision jump means the FP are
        misaligned-but-real elements; no jump means hallucinated FP.
        """
        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
        num_classes = len(self.map_classes)
        num_rings = len(ring_edges) - 1
        num_t = len(thresholds)
        tolerances = tuple(int(d) for d in (tolerances or ()) if int(d) > 0)
        tp = torch.zeros(num_rings, num_classes, num_t)
        fp = torch.zeros(num_rings, num_classes, num_t)
        fn = torch.zeros(num_rings, num_classes, num_t)
        # Alignment-tolerant counts per tolerance radius:
        #   tp_tol: predicted cells falling inside the d-px-dilated GT
        #   cov_tol: GT cells covered by the d-px-dilated prediction
        tp_tol = {d: torch.zeros(num_rings, num_classes, num_t)
                  for d in tolerances}
        cov_tol = {d: torch.zeros(num_rings, num_classes, num_t)
                   for d in tolerances}
        # Coverage accumulators (starvation precheck): is the model even
        # predicting positives in the far rings vs what the GT requires?
        gt_pos = torch.zeros(num_rings, num_classes)        # GT-positive cells
        pred_pos = torch.zeros(num_rings, num_classes)      # pred>=0.35 cells
        prob_sum = torch.zeros(num_rings, num_classes)      # sum of pred prob
        ring_cells = torch.zeros(num_rings)                 # cells per ring
        ring_masks = None

        for result in results:
            pred = torch.as_tensor(result['masks_bev']).detach().float()
            label = torch.as_tensor(result['gt_masks_bev']).detach().bool()
            h, w = pred.shape[-2:]
            if ring_masks is None:
                ring_masks = self._build_ring_masks(h, w, map_range, ring_edges)

            pred_bin_hw = (
                pred[:, None] >= thresholds.view(1, -1, 1, 1))  # (C,T,H,W)
            dil_gt = {}
            dil_pred = {}
            for d in tolerances:
                kernel = 2 * d + 1
                dil_gt[d] = F.max_pool2d(
                    label[None].float(), kernel, stride=1,
                    padding=d)[0].bool().reshape(num_classes, -1)
                dil_pred[d] = F.max_pool2d(
                    pred_bin_hw.float().reshape(1, num_classes * num_t, h, w),
                    kernel, stride=1,
                    padding=d)[0].bool().reshape(num_classes, num_t, -1)

            pred = pred.reshape(num_classes, -1)
            label = label.reshape(num_classes, -1)
            pred_bin = pred_bin_hw.reshape(num_classes, num_t, -1)  # (C,T,HW)
            for ri, mask in enumerate(ring_masks):
                pb = pred_bin[:, :, mask]                   # (C, T, ring)
                lb = label[:, None, mask]                   # (C, 1, ring)
                tp[ri] += (pb & lb).sum(dim=2)
                fp[ri] += (pb & ~lb).sum(dim=2)
                fn[ri] += (~pb & lb).sum(dim=2)
                for d in tolerances:
                    tp_tol[d][ri] += (pb & dil_gt[d][:, None, mask]).sum(dim=2)
                    cov_tol[d][ri] += (dil_pred[d][:, :, mask] & lb).sum(dim=2)
                gt_pos[ri] += label[:, mask].sum(dim=1)
                pred_pos[ri] += (pred[:, mask] >= 0.35).sum(dim=1)
                prob_sum[ri] += pred[:, mask].sum(dim=1)
                ring_cells[ri] += int(mask.sum())

        # Append an "all rings" aggregate row so the whole-grid tolerant
        # precision/recall come out of the same pass.
        def _with_total(t):
            return torch.cat([t, t.sum(dim=0, keepdim=True)])

        tp, fp, fn = _with_total(tp), _with_total(fp), _with_total(fn)
        tp_tol = {d: _with_total(v) for d, v in tp_tol.items()}
        cov_tol = {d: _with_total(v) for d, v in cov_tol.items()}
        gt_pos, pred_pos = _with_total(gt_pos), _with_total(pred_pos)
        prob_sum, ring_cells = _with_total(prob_sum), _with_total(ring_cells)
        tags = [f'map_ring{lo:g}-{hi:g}m'
                for lo, hi in zip(ring_edges[:-1], ring_edges[1:])]
        tags.append('map_ring_all')

        ious = tp / (tp + fp + fn + 1e-7)                   # (R+1, C, T)
        metrics = {}
        for ri, tag in enumerate(tags):
            per_class_max = ious[ri].max(dim=1).values       # (C,)
            best_ti = ious[ri].argmax(dim=1)                 # (C,)
            cells = ring_cells[ri].clamp(min=1.0)
            for ci, name in enumerate(self.map_classes):
                ti = best_ti[ci]
                tp_b, fp_b, fn_b = tp[ri, ci, ti], fp[ri, ci, ti], fn[ri, ci, ti]
                metrics[f'{tag}/{name}/iou@max'] = per_class_max[ci].item()
                metrics[f'{tag}/{name}/best_thr'] = thresholds[ti].item()
                metrics[f'{tag}/{name}/prec@best'] = (
                    tp_b / (tp_b + fp_b + 1e-7)).item()
                metrics[f'{tag}/{name}/rec@best'] = (
                    tp_b / (tp_b + fn_b + 1e-7)).item()
                for d in tolerances:
                    metrics[f'{tag}/{name}/prec_tol{d}px@best'] = (
                        tp_tol[d][ri, ci, ti] / (tp_b + fp_b + 1e-7)).item()
                    metrics[f'{tag}/{name}/rec_tol{d}px@best'] = (
                        cov_tol[d][ri, ci, ti] / (tp_b + fn_b + 1e-7)).item()
                metrics[f'{tag}/{name}/gt_pos_rate'] = (gt_pos[ri, ci] / cells).item()
                metrics[f'{tag}/{name}/pred_pos_rate@0.35'] = (pred_pos[ri, ci] / cells).item()
                metrics[f'{tag}/{name}/mean_prob'] = (prob_sum[ri, ci] / cells).item()
            metrics[f'{tag}/mean/iou@max'] = per_class_max.mean().item()
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

        map_show_dir = eval_kwargs.get('map_show_dir', None)
        if has_map and map_show_dir is not None:
            self.save_map_results(results, map_show_dir)

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

            # Optional per-distance-ring breakdown on the native eval grid, to
            # localise where the map gap lives. Pass via --eval-options:
            #   map_ring_edges=0,20,40,50,71 map_ring_range=50
            #   map_ring_tolerances=1,2   (px; alignment-tolerant prec/rec)
            map_ring_edges = eval_kwargs.get('map_ring_edges', None)
            if map_ring_edges is not None:
                if isinstance(map_ring_edges, str):
                    map_ring_edges = [float(x) for x in map_ring_edges.split(',')]
                map_ring_edges = tuple(float(x) for x in map_ring_edges)
                map_ring_range = float(eval_kwargs.get('map_ring_range', 50.0))
                map_ring_tol = eval_kwargs.get('map_ring_tolerances', (1, 2))
                if isinstance(map_ring_tol, str):
                    map_ring_tol = [
                        int(float(x)) for x in map_ring_tol.split(',') if x]
                elif isinstance(map_ring_tol, (int, float)):
                    map_ring_tol = [int(map_ring_tol)]
                eval_results.update(self.evaluate_map_rings(
                    results, map_range=map_ring_range,
                    ring_edges=map_ring_edges,
                    tolerances=tuple(int(x) for x in map_ring_tol)))

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
