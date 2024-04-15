# :sparkles: MS2A: Memory Storage-to-Adaptation for Cross-domain Few-annotation Object Detection [[Project Page]](https://ms2a-cfod.github.io/)
This repository contains the code for reproducing the paper: **[MS2A: Memory Storage-to-Adaptation for Cross-domain Few-annotation Object Detection](https://)**
![Demo](./figures/figure%202-2-rf.jpg)

## :computer: MS2A work
- We propose a novel memory storage-to-adaptation mechanism to learn the prior knowledge and transfer it to feature alignment adaptively. To the best of our knowledge, this is the first work to extract the prior knowledge of unlabeled target data to address the CFOD task.
- We construct a `new challenging benchmark` of industrial scenarios, that comprises complex background and large differences between source and target domains.
- Experiments show that the proposed MS2A outperforms state-of-the-art methods on both public datasets and the constructed industrial dataset. In particular, MS2A exceeds the `state-of-the-art` method by 10.4\% on the challenging industrial dataset for the 10-annotation setting.

## :bell: About training and testing
### :one: train from scratch
```shell
Stage I - Memory Storage

1. pretrain base detector
./tools/dist_train.sh ${config_file} ${gpu_number}
e.g: ./tools/dist_train.sh ./configs/yolox/yolox_fs_x_640x640_50e_18_base.py 1

2. extract and cluster the prior knowledge
python extract_fea_fewshot.py \
 --depart ${dataset name} \
 --img ${image dir} \
 --config ${config file} \
 --checkpoint ${checkpoint file} \
 --out_path ${save dir} \
 --batch_size 8 \
 --dim ${128/256 large:256}
e.g: 
python extract_fea_fewshot.py \
 --depart Indus_T1 \
 --img /nas/public_data/foggy_cityscapes_COCO_format/train2014 \
 --config ./configs/yolox/yolox_fs_x_640x640_50e_cityscapes_base.py \
 --checkpoint ./work_dirs/yolox_fs_x_640x640_50e_cityscapes_base/best_bbox_mAP_epoch_70.pth \
 --out_path /path/to/save/ckpt \
 --batch_size 8 \
 --dim 320

python knn_cuda.py \
 --fea_root_path ${last step saved dir} \
 --out_path ${default same as fea_root_path} \
 --n_clusters ${default 100} \
 --depart ${cluster name}
e.g:
python knn_cuda.py \
 --fea_root_path /path/to/save/ckpt/ \
 --out_path /path/to/save/cluster/results/ \
 --n_clusters 100 \
 --depart Indus_T1

Stage II - Memory Adaptation

1. Source Training
./tools/dist_train.sh ${config_file} ${gpu_number}
e.g: ./tools/dist_train.sh configs/yolox/yolox_fs_x_640x640_50e_cityscapes_pretrain_mom.py 1

2. Target Training
./tools/dist_train.sh ${config_file} ${gpu_number}
e.g: ./tools/dist_train.sh configs/yolox/yolox_fs_x_640x640_50e_cityscapes_pretrain_mom_ft.py 1
```
### :two: eval
```shell
python tools/test.py ${config file} ${checkpoint file} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
e.g:
python tools/test.py \
 ./configs/yolox/yolox_fs_x_640x640_50e_cityscapes_pretrain_mom_ft.py \
 ./work_dirs/yolox_fs_x_640x640_50e_cityscapes_pretrain_mom_ft/best_0_bbox_mAP_epoch_20.pth \
 --eval bbox
```
### :three: inference
```shell
python tools/inference.py ${config file} ${checkpoint file} ${save dir}
e.g:
python tools/test.py \
 ./configs/yolox/yolox_fs_x_640x640_50e_cityscapes_pretrain_mom_ft.py \
 ./work_dirs/yolox_fs_x_640x640_50e_cityscapes_pretrain_mom_ft/best_0_bbox_mAP_epoch_20.pth \
 ./path/to/save
```

## Thanks
Thanks to the public repos: [mmdetection](https://github.com/open-mmlab/mmdetection) for providing the base code.