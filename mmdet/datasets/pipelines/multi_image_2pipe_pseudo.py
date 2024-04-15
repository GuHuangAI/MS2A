#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：mmdetection 
@File ：MultiImage2PipePseudoDataset.py
@Author ：Xiawei
@Date ：2022/12/6 16:45

'''

import bisect
import collections
import copy
import math
import random
from collections import defaultdict

import numpy as np
from mmcv.utils import build_from_cfg, print_log
from mmdet.datasets.builder import DATASETS, PIPELINES
import copy
from mmdet.datasets.pipelines.tools import visual,visualImgList

from mmdet.datasets.pipelines import RandomAugSingle
import random
import torch
from mmdet.datasets.datasetwrapper.multi_image_base import MultiImageBase



# 生成数据 + 无监督数据 + 可更新 + 类别均衡
@DATASETS.register_module()
class MultiImage2PipePseudoDataset(MultiImageBase):
    def __init__(self,
                 dataset,
                 dataset_gen,
                 dataset_gen_dict,  # 生成的动态set
                 labeldata_rait=None,
                 pipeline=None,
                 final_pipeline=None,
                 dynamic_scale=None,
                 skip_type_keys=None,
                 max_refetch=15,
                 visualization=False,
                 visualization_dir='visual',
                 random_mosic_rait=0.5,
                 ratio_type_keys=['Mosaic', 'RandomAffine', 'MixUp'],
                 unlable_retimes=0,
                 aug_type='seq_nocrop',
                 strong_aug=True,
                 close_aug=False,
                 label_aug_type='default',
                 ):
        super(MultiImage2PipePseudoDataset, self).__init__(
            dataset, pipeline,dynamic_scale,skip_type_keys,max_refetch,visualization,visualization_dir,ratio_type_keys)
        self.dataset_gen_dict = dataset_gen_dict
        self.dataset_gen = dataset_gen
        self.visualization = visualization
        self.random_mosic_rait = random_mosic_rait
        self.strong_aug = strong_aug
        self.close_aug = close_aug
        self.labeldata_rait = labeldata_rait
        self.label_aug_type = label_aug_type

        self.randonaugul = RandomAugSingle(aug_type=aug_type)
        self.randonaugul_default = RandomAugSingle(aug_type='weak')

        self.unlable_retimes = unlable_retimes
        self.final_pipelines = []
        for transform in final_pipeline:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.final_pipelines.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self.num_samples = len(self.dataset)
        print('self.num_samples',self.num_samples)

    def __len__(self):
        return self.num_samples

    def _getitem_notNone(self, idx,dataset,strong_aug=True):
        results, transformtype, origindata = self._getitem_noNone(idx, dataset,strong_aug)
        if results is None:
            print('random   sample None data  using new {}  '.format( transformtype))
            # for k,v in origindata.items():
            #     if k not in ('img','image'):
            #         print(k,v)
            return self._getitem_notNone(idx + 1,dataset,strong_aug)
        return results

    def __getitem__(self, idx):
        results = self._getitem_notNone(idx, self.dataset) # 有监督
        results['sources'] = 0
        r1 = copy.deepcopy(results)

        if self.label_aug_type == 'default':
            r2 = self.randonaugul(results)
        else: # default
            r2 = self.randonaugul_default(results)

        if self.visualization:
            visualImgList([r1,r2], self.randonaugul, "aug_ucl_label")

        aug_pair_list = []
        for i in range(self.unlable_retimes): # 采样 3次无监督
            originresults = self._getitem_notNone(random.randint(0,len(self.dataset_gen)-1),self.dataset_gen,self.strong_aug)
            nolr1 = copy.deepcopy(originresults)
            nolr2 = self.randonaugul(originresults)
            aug_pair_list.append((nolr1,nolr2))

        if self.visualization:
            for (v1,v2) in aug_pair_list:
                visualImgList([v1,v2], self.randonaugul, "aug_ucl")

        tlist = [r1,r2]
        for (v1,v2) in aug_pair_list:
            v1['sources'] = 1
            v2['sources'] = 1
            tlist.append(v1)
            tlist.append(v2)

        resultinfos = [] # merge all data
        for updated_results in tlist:
            for t in self.final_pipelines:
                updated_results = t(updated_results)  #
            resultinfos.append(updated_results)
        return resultinfos

    # 1: 1v1 的结果
    def update_ann_file(self,ann_file,img_prefix=''):
        from mmdet.datasets import build_dataset
        print('New data ann_file .',ann_file)
        self.dataset_gen_dict['dataset']['ann_file'] = ann_file
        if len(img_prefix) > 0:
            self.dataset_gen_dict['dataset']['img_prefix'] = img_prefix
        cfg_copy = self.dataset_gen_dict.copy()
        # cfg_copy.pop('type')

        if self.dataset_gen is not None:
            print('Old data len .', len(self.dataset_gen))
            if hasattr(self.dataset_gen, 'total_ann_ids'):
                print('Old data total_ann_ids .', self.dataset_gen.total_ann_ids)
            del self.dataset_gen
        self.dataset_gen = build_dataset(cfg_copy)
        print('finished flush dataset.')
        print('New data len .',len(self.dataset_gen))
        if hasattr(self.dataset_gen,'total_ann_ids'):
            print('New data total_ann_ids .', self.dataset_gen.total_ann_ids)
        # print('finished flush dataset.')

    def _getitem_noNone(self, idx, dataset,strong_aug=True):
        # 一个数据集尝试多次
        transformtype = None
        origindata = None
        for i in range(self.max_refetch): # 改动长度没法试试更新step长度，无法对最后的数据读取。这里加入随机打乱数据集
            idx = random.randint(0,self.num_samples-1)
            results,transformtype , origindata = self._getitem(idx, dataset, strong_aug)
            if results is not None:
                return results,transformtype , origindata
        return None,transformtype, origindata

    def _getitem(self,idx,dataset,strong_aug=True):
        if idx >= len(dataset):
            idx = random.randint(0,len(dataset)-1)
        results = copy.deepcopy(dataset[idx])
        random_mosic = random.random()
        for id, (transform, transform_type) in enumerate(zip(self.pipeline, self.pipeline_types)):
            if results == None:
                return None,self.pipeline_types[id-1],dataset[idx]
            if random_mosic > self.random_mosic_rait or not strong_aug: # r=0.2， 80%概率会pass掉，20%概率继续
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
                return None, self.pipeline_types[id],dataset[idx]

            if 'mix_results' in results:
                results.pop('mix_results')

            # show processed img
            if self.visualization:
                visual(results, transform, str(id) + "after",visualization_dir=self.visualization_dir)

        return results, self.pipeline_types[-1],dataset[idx]