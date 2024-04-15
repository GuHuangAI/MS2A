#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bisect
import collections
import copy
import math
import random
from collections import defaultdict

import numpy as np
from mmcv.utils import build_from_cfg, print_log
from ..builder import DATASETS, PIPELINES

# from mmdet.datasets.pipelines.tools import visual,visualImgList

# from mmdet.datasets.pipelines import RandomAugUL
import random
import torch
from ..dataset_wrappers import MultiImageMixDataset



@DATASETS.register_module()
class MultiImageBase(MultiImageMixDataset):
    # 实现基本的get函数
    def __init__(self,
                 dataset,
                 pipeline,
                 dynamic_scale=None,
                 skip_type_keys=None,
                 max_refetch=15, # 从多少个里组合mosaic
                 visualization=False,
                 visualization_dir='./visualization',
        ratio_type_keys = ['Mosaic', 'RandomAffine', 'MixUp']
    ):
        super(MultiImageBase, self).__init__(
            dataset, pipeline,dynamic_scale,skip_type_keys,max_refetch,visualization,visualization_dir)
        self.ratio_type_keys = ratio_type_keys

    def _getitem_noNone(self, idx, dataset):
        # 一个数据集尝试多次
        transformtype = None
        for i in range(self.max_refetch):
            results,transformtype = self._getitem(idx, dataset)
            if results is not None:
                return results,transformtype
        return None,transformtype

    def _getitem(self,idx,dataset):
        if idx >= len(dataset):
            idx = random.randint(0,len(dataset)-1)
        results = copy.deepcopy(dataset[idx])
        random_mosic = random.random()
        for id, (transform, transform_type) in enumerate(zip(self.pipeline, self.pipeline_types)):
            if results == None:
                return None,self.pipeline_types[id-1]
            if random_mosic > self.random_mosic_rait: # r=0.2， 80%概率会pass掉，20%概率继续
                if transform_type in self.ratio_type_keys:
                    continue # 随机不要mosaic

            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue
            if hasattr(transform, 'get_indexes'):
                for i in range(self.max_refetch):
                    # Make sure the results passed the loading pipeline
                    # of the original dataset is not None.
                    indexes = transform.get_indexes(dataset) # 随机选一个
                    if not isinstance(indexes, collections.abc.Sequence):
                        indexes = [indexes]
                    mix_results = [
                        copy.deepcopy(dataset[index]) for index in indexes
                    ]
                    if None not in mix_results:
                        results['mix_results'] = mix_results
                        break
                else:
                    raise RuntimeError(
                        'The loading pipeline of the original dataset'
                        ' always return None. Please check the correctness '
                        'of the dataset and its pipeline.')

            # 第一次显示输入
            if self.visualization and id == 0:
                visual(results, transform, str(id) + "before",visualization_dir=self.visualization_dir)

            for i in range(self.max_refetch): # 可以试多次
                updated_results = transform(copy.deepcopy(results)) # 这里就aug好了
                if updated_results is not None:
                    results = updated_results
                    break
            else:
                if self.visualization:
                    print('max_refetch get  None data ')
                return None, self.pipeline_types[id]

            if 'mix_results' in results:
                results.pop('mix_results')

            # show processed img
            if self.visualization:
                visual(results, transform, str(id) + "after",visualization_dir=self.visualization_dir)

        return results, self.pipeline_types[-1]
