# Copyright (c) OpenMMLab. All rights reserved.
from mmpretrain.utils.dependency import WITH_MULTIMODAL
from .base_dataset import BaseDataset
from .builder import build_dataset
from .custom import CustomDataset
from .dataset_wrappers import KFoldDataset
from .multi_label import MultiLabelDataset
from .multi_task import MultiTaskDataset
from .samplers import *  # noqa: F401,F403
from .transforms import *  # noqa: F401,F403
from .apple import Apple
from .citrus import Citrus
from .plant import Plant
from .rice import Rice
from .tomato import Tomato
from .wheat import Wheat

__all__ = [
    'BaseDataset', 'Apple', 'Citrus', 'Plant', 'Rice', 'Tomato',
    'Wheat', 'MultiLabelDataset', 'CustomDataset',
    'MultiTaskDataset', 'build_dataset'
]
