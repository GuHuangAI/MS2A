<!--
 * @Date: 2024-04-15 17:37:57
 * @LastEditors: Shilong Zou
 * @LastEditTime: 2024-04-15 20:50:19
 * @FilePath: \rw5yzo2qd:\Personal Files\few_shot_code\README.md
-->
## :sparkles: MS2A: Memory Storage-to-Adaptation for Cross-domain Few-annotation Object Detection

### :computer: MS2A work
- We propose a novel memory storage-to-adaptation mechanism to learn the prior knowledge and transfer it to feature alignment adaptively. To the best of our knowledge, this is the first work to extract the prior knowledge of unlabeled target data to address the CFOD task.
- We construct a `new challenging benchmark` of industrial scenarios, that comprises complex background and large differences between source and target domains.
- Experiments show that the proposed MS2A outperforms state-of-the-art methods on both public datasets and the constructed industrial dataset. In particular, MS2A exceeds the `state-of-the-art` method by 10.4\% on the challenging industrial dataset for the 10-annotation setting.

### :bell: How to start training
##### :one: train
```shell
Stage I - Memory Storage

1. pretrain base detector
./tools/dist_train.sh ${config_file} ${gpu_number}
e.g: ./tools/dist_train.sh ./configs/yolox/yolox_fs_x_640x640_50e_18_base.py 1
2. extract and cluster the prior knowledge
python extract_fea_fewshot.py --depart ${dataset name} --img ${image dir} --config ${config file} --checkpoint ${checkpoint file} --out_path ${save dir} --batch_size 8 --dim ${128/256 large:256}
python knn_cuda.py --fea_root_path ${last step saved dir} --out_path ${default same as fea_root_path} --n_clusters ${default 100} –depart ${cluster name}

Stage II - Memory Adaptation

1. Source Training
./tools/dist_train.sh ${config_file} ${gpu_number}
2. Target Training
./tools/dist_train.sh ${config_file} ${gpu_number}
```
##### :two: eval
```shell
python tools/test.py ${config file} ${checkpoint file} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
```
##### :three: inference
```shell
python tools/inference.py ${config file} ${checkpoint file} ${save dir}
```