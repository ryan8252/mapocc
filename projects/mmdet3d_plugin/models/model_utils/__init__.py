from .depthnet import DepthNet
from .maestro_cpg import *
from .maestro_spa import *
from .maestro_tsfg import *

__all__ = [
    'DepthNet',
    'MAESTROClasswisePrototypeGenerator',
    'MAESTROOnewayScenePrototypeAggregator',
    'MAESTROTaskSpecificFeatureGenerator',
]
