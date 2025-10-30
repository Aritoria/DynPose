# Copyright (c) OpenMMLab. All rights reserved.
from .ae_head import AssociativeEmbeddingHead
from .cid_head import CIDHead
from .cpm_head import CPMHead
from .heatmap_head import HeatmapHead
from .internet_head import InternetHead
from .mspn_head import MSPNHead
from .vipnas_head import ViPNASHead
from .heatmap_head_imageIn import HeatmapHeadImageIn
from .heatmap_head_imageIn_multiin import HeatmapHeadImageMultiIn
from .twohead import TwoHead
from .dyheatmap_head import DYHeatmapHead
from .moe_head import MoEHead
__all__ = [
    'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead',
    'AssociativeEmbeddingHead', 'CIDHead', 'InternetHead','HeatmapHeadImageIn','HeatmapHeadImageMultiIn',
    'TwoHead','DYHeatmapHead','MoEHead'
]
