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

from mmdet.datasets.pipelines.tools import visual,visualImgList

import random
import torch
from .multi_image_base import MultiImageBase



@DATASETS.register_module()
class MultiImageMultilSourceDataset(MultiImageBase):
       #源数据和目标数据都采样
       #都有比例控制， 最内部根据原采样， 增广的时候混合
       #带 source
    def __init__(self,
                 dataset_list,      #  数据的list
                 dataset_raits,     # 比例
                 dataset_sources,   # 来源
                 pipeline,
                 dynamic_scale=None,
                 skip_type_keys=None,
                 max_refetch=5,
                 visualization=False,
                 random_mosic_rait=0.5,
                 need_source=True,
                 visualization_dir='visual',
                 ratio_type_keys=['Mosaic', 'RandomAffine', 'MixUp']
                 ):
        super(MultiImageMultilSourceDataset, self).__init__(
            dataset_list[0], pipeline,dynamic_scale,skip_type_keys,max_refetch,visualization_dir,ratio_type_keys)
        self.dataset_list = dataset_list
        self.dataset_raits = dataset_raits
        self.dataset_sources = dataset_sources
        self.visualization = visualization
        self.num_samples = len(self.dataset_list[0])
        self.need_source = need_source
        self.random_mosic_rait = random_mosic_rait

        for da,rait,source in zip(self.dataset_list,self.dataset_raits,self.dataset_sources):
            print('setlen {} - rait{} - source {}',len(da),rait,source)

    def __getitem__(self, idx):
        rd = random.random()
        results = None
        dataid = 0
        for dataset, rait, source in zip(self.dataset_list, self.dataset_raits, self.dataset_sources):
            if len(dataset) <= 0:
                print('null dataset ',dataset)
                continue
            if (rd < rait):
                results,transformtype = self._getitem_noNone(idx,dataset)
                if results is None:
                    print('random {} sample None data from {} ，using new {}  '.format(rd, dataid,transformtype))
                    return self.__getitem__(idx + 1)
                if self.need_source:
                    results['sources'] = int(source)
                break
            else:
                rd = rd - rait
                dataid += 1
        return results


