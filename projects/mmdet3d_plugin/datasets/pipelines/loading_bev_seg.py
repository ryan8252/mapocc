import numpy as np
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

        self.dataset_root = dataset_root
        self._map_location = None   # currently loaded location
        self._map_obj = None        # single NuScenesMap kept in RAM

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

        # Use rotation + scale only — no flip.
        # get_map_mask accepts a scalar angle, so reflections cannot be encoded
        # in patch_angle (a flip would appear as a 180° rotation, causing
        # flip_x + flip_y instead of just flip_x/flip_y). Flips are applied
        # post-rasterization in __call__ to match LoadOccGTFromFile's behaviour.
        rotate_bda = float(results.get('rotate_bda', 0.0))
        scale_bda = float(results.get('scale_bda', 1.0))
        c, s = np.cos(-rotate_bda), np.sin(-rotate_bda)
        rot_mat = np.array(
            [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        lidar_aug = np.eye(4, dtype=np.float32)
        lidar_aug[:3, :3] = scale_bda * rot_mat

        aug_lidar2global = ego2global @ lidar2ego @ np.linalg.inv(lidar_aug)
        return aug_lidar2global

    def _build_layer_mapping(self):
        mappings = {}
        for name in self.classes:
            if name in ('drivable_area', 'drivable_area*'):
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
        if location != self._map_location:
            self._map_obj = NuScenesMap(self.dataset_root, location)
            self._map_location = location
        masks = self._map_obj.get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # get_map_mask returns (C, canvas_h, canvas_w) = (C, y_cells, x_cells).
        # Transpose to (C, x_cells, y_cells) before boolean assignment.
        # Because labels is created with canvas_size shape (y_cells, x_cells), and the grid
        # is symmetric (x_cells == y_cells for nuScenes), numpy silently applies the boolean
        # mask without error but physically swaps x and y indices in labels.
        masks = masks.transpose(0, 2, 1).astype(np.bool_)

        labels = np.zeros((len(self.classes), *self.canvas_size), dtype=np.int64)
        for class_index, class_name in enumerate(self.classes):
            for layer_name in mappings[class_name]:
                mask_index = layer_names.index(layer_name)
                labels[class_index, masks[mask_index]] = 1

        # The two transposes cancel the index swap, leaving labels in (C, x_cells, y_cells).
        # The [:, :, ::-1] corrects the y-axis direction: NuScenes row 0 = y_max (North = top),
        # flipping gives y_idx=0 = y_min to match voxel_semantics[x_idx, y_idx, z_idx].
        labels = labels.transpose(0, 2, 1)[:, :, ::-1]

        # Post-rasterization flip to match LoadOccGTFromFile's BDA flip:
        #   flip_dx -> torch.flip(semantics, [0])  => labels[:, ::-1, :]
        #   flip_dy -> torch.flip(semantics, [1])  => labels[:, :, ::-1]
        # get_map_mask only accepts a scalar angle, so reflections must be applied here.
        if results.get('flip_dx', False):
            labels = labels[:, ::-1, :]
        if results.get('flip_dy', False):
            labels = labels[:, :, ::-1]

        results['gt_masks_bev'] = np.ascontiguousarray(labels)
        return results
