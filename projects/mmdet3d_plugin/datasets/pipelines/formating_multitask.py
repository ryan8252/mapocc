from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.builder import PIPELINES

from .formating import DefaultFormatBundle3D


@PIPELINES.register_module()
class MultitaskFormatBundle3D(DefaultFormatBundle3D):
    """Format ProtoOcc occupancy fields plus BEV segmentation supervision."""

    def __call__(self, results):
        results = super(MultitaskFormatBundle3D, self).__call__(results)

        if 'gt_masks_bev' in results:
            results['gt_masks_bev'] = DC(
                to_tensor(results['gt_masks_bev']),
                stack=True,
            )

        return results
