from mmdet.apis import init_detector
import mmcv
import numpy as np
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import torch
import os
import json
from tqdm import tqdm


def prob2entropy_ln(label):
    label = np.array(label)
    eps = 1e-8
    return - np.log2(label + eps)


def get_histogram_two( test):
    bins1 = np.array([0.01, 0.109, 0.208, 0.307, 0.406, 0.505, 0.604, 0.703, 0.802, 0.901, 1])

    hist1, bin1 = np.histogram(test, bins=bins1)
    # hist2, bin2 = np.histogram(val, bins=bin1)

    p1 = hist1 / sum(hist1)
    # p2 = hist2 / sum(hist2)
    # print(p1, bin1)
    # print(p2, bin2)

    return p1


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    #cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    #test_pipeline = Compose(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.test_pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None, env_id=4)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results
        

# 指定模型的配置文件和 checkpoint 文件路径
config_file = "/home/ml/code/mmdetection/configs/yolox/yolox_l_8x8_300e_bigmodel_bigsecond_fewshot_pretrain_xingbang_ft.py"
checkpoint_file = "/home/ml/code/mmdetection/work_dirs/yolox_l_8x8_300e_bigmodel_bigsecond_fewshot_pretrain_xingbang_ft/best_bbox_mAP_epoch_10.pth"

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并展示结果

img_path = "/nfs/clean_test/image/bigsecond/xingbang"
img_list = [os.path.join(img_path, dirlist, image) for dirlist in os.listdir(img_path) for image in os.listdir(os.path.join(img_path, dirlist)) if image.endswith('.jpg')]

print('#'*10, 'There are {} images'.format(len(img_list)))


out_path = '/nfs/datamonitor/monitor_0406'

if not os.path.exists(out_path):
    os.makedirs(out_path)

res = torch.tensor([[0]])

for id, img in tqdm(enumerate(img_list, 1)):
    if id == 1:
        res = inference_detector(model, img)
        tmp_dic = {'model': 'xingbang_fs', 'model_type':'Detection', 'results': res[0].tolist()}
        print(img)
        json.dump(tmp_dic, open(os.path.join(out_path, '{}_{}.json'.format(img.split('/')[-2], os.path.basename(img).split('.')[0])), 'w'))
        
    result = inference_detector(model, img)
    
    tmp_dic = {'model': 'xingbang_fs', 'model_type':'Detection', 'results': result[0].tolist()}
    json.dump(tmp_dic, open(os.path.join(out_path, '{}_{}.json'.format(img.split('/')[-2], os.path.basename(img).split('.')[0])), 'w'))

    #res = torch.concat([res, result], dim=0)

''' 
res = res[:, 4]
prob = prob2entropy_ln(res)
list2 = get_histogram_two(prob)

save_dic = {'model': 'xingbang_fs', 'val_data': list2.tolist()}
json.dump(save_dic, open(os.path.join(out_path, 'modelout.json'), 'w'))
   
img = '/nfs/testset/xingbang_bigsecond_0403_COCO_format/微信图片_20230405233511.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
result = inference_detector(model, img)
print(result)
# 在一个新的窗口中将结果可视化
model.show_result(img, result)
# 或者将可视化结果保存为图片
model.show_result(img, result, out_file='_20230405233511.jpg', bbox_color='red', thickness=4, text_color='red')
'''