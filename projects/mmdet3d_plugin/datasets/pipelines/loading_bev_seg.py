import numpy as np
import torch
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion

from mmdet3d.datasets.builder import PIPELINES


LOCATIONS = (
    'boston-seaport',
    'singapore-hollandvillage',
    'singapore-onenorth',
    'singapore-queenstown',
)


@PIPELINES.register_module()
class LoadBEVSegmentation(object):
    """Generate BEVFusion-style HD map supervision on the fly.

    The map patch is rasterized from the official nuScenes map API using the
    augmented LiDAR frame pose so that the GT stays aligned with ProtoOcc's
    BEV augmentation (`bda_rot`).
    """

    def __init__(self, dataset_root, xbound, ybound, classes):
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])

        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = tuple(classes)

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def _build_lidar2global(self, results):
        curr = results['curr']

        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego[:3, :3] = Quaternion(
            curr['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = curr['lidar2ego_translation']

        ego2global = np.eye(4, dtype=np.float32)
        ego2global[:3, :3] = Quaternion(
            curr['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = curr['ego2global_translation']

        bda_rot = results['img_inputs'][6]
        if isinstance(bda_rot, torch.Tensor):
            bda_rot = bda_rot.detach().cpu().numpy()

        lidar_aug = np.eye(4, dtype=np.float32)
        lidar_aug[:3, :3] = bda_rot

        aug_lidar2global = ego2global @ lidar2ego @ np.linalg.inv(lidar_aug)
        return aug_lidar2global

    def _build_layer_mapping(self):
        mappings = {}
        for name in self.classes:
            if name == 'drivable_area*':
                mappings[name] = ['road_segment', 'lane']
            elif name == 'divider':
                mappings[name] = ['road_divider', 'lane_divider']
            else:
                mappings[name] = [name]
        return mappings

    def __call__(self, results):
        if 'location' not in results:
            raise KeyError(
                'LoadBEVSegmentation requires `location` in results. '
                'Please use the multitask nuScenes dataset that injects '
                'scene_token -> location lookup results.'
            )

        lidar2global = self._build_lidar2global(results)
        map_pose = lidar2global[:2, 3]
        patch_box = (
            map_pose[0],
            map_pose[1],
            self.patch_size[0],
            self.patch_size[1],
        )

        rotation = lidar2global[:3, :3]
        direction = rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        patch_angle = np.arctan2(direction[1], direction[0]) / np.pi * 180.0

        mappings = self._build_layer_mapping()
        layer_names = []
        for names in mappings.values():
            layer_names.extend(names)
        layer_names = list(dict.fromkeys(layer_names))

        location = results['location']
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        masks = masks.transpose(0, 2, 1).astype(np.bool_)

        labels = np.zeros((len(self.classes), *self.canvas_size), dtype=np.int64)
        for class_index, class_name in enumerate(self.classes):
            for layer_name in mappings[class_name]:
                mask_index = layer_names.index(layer_name)
                labels[class_index, masks[mask_index]] = 1

        results['gt_masks_bev'] = labels
        return results
