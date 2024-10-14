# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine import get_file_backend, list_from_file

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import CITRUS_CATEGORIES


@DATASETS.register_module()
class Citrus(BaseDataset):
    """
        ├── train
        │   ├── class_x
        │   │   ├── xx1.jpg
        │   │   ├── xx2.jpg
        │   │   └── ...
        │   ├── class_y
        │   │   ├── yy1.jpg
        │   │   ├── yy2.jpg
        │   │   └── ...
        │   └── ...
        ├── val
        │   ├── class_x
        │   │   ├── xx1.jpg
        │   │   ├── xx2.jpg
        │   │   └── ...
        │   ├── class_y
        │   │   ├── yy1.jpg
        │   │   ├── yy2.jpg
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── train.txt
        │   └── val.txt
        └── ....

    Args:
        data_root (str): The root directory for dataset.
        split (str, optional): The dataset split, supports "train" and "val".
            Default to "train".

    """  # noqa: E501

    METAINFO = {'classes': CITRUS_CATEGORIES}

    def __init__(self, data_root: str, split: str = 'train', **kwargs):

        splits = ['train', 'val']
        assert split in splits, \
            f"The split must be one of {splits}, but get '{split}'"
        self.split = split

        self.backend = get_file_backend(data_root, enable_singleton=True)
        if split == 'train':
            ann_file = self.backend.join_path('meta', 'train.txt')
            data_prefix = 'train'
        else:
            ann_file = self.backend.join_path('meta', 'val.txt')
            data_prefix = 'val'

        test_mode = split == 'val'

        super(Citrus, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            test_mode=test_mode,
            data_prefix=data_prefix,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""

        pairs = list_from_file(self.ann_file)
        data_list = []
        for pair in pairs:
            class_name, img_name = pair.split('/')
            # img_name = f'{img_name}.jpg'
            img_path = self.backend.join_path(self.img_prefix, class_name,
                                              img_name)
            gt_label = self.METAINFO['classes'].index(class_name)
            info = dict(img_path=img_path, gt_label=gt_label)
            data_list.append(info)
        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
