import asyncio
import os.path
from argparse import ArgumentParser
import json
# from mmdet.apis import (async_inference_detector, inference_detector,
#                         init_detector, show_result_pyplot)
import warnings
from pathlib import Path
from tqdm import tqdm
import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmcv.parallel import collate

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='img file dir or json', default='/nfs/testset/xingbang_bigsecond_0403_COCO_format/train2017/000000000001.jpg')
    parser.add_argument('--img_root', help='if the input of --img is a json file, this must not be none', default=None)
    parser.add_argument('--config', help='Config file',
                        default="/home/ml/code/mmdetection/configs/yolox/yolox_l_8x8_300e_bigmodel_bigsecond_fewshot_pretrain_xingbang_ft.py")
    parser.add_argument('--checkpoint', help='Checkpoint file',
                        default="/home/ml/code/mmdetection/work_dirs/yolox_l_8x8_300e_bigmodel_bigsecond_fewshot_pretrain_xingbang_ft/best_bbox_mAP_epoch_10.pth")

    parser.add_argument('--out_path', default='/home/ml/code/mmdetection/fewshot_fea', help='Path to output file')
    #parser.add_argument('--batch_size', default=8, type=int, help='test batch size')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmcv.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if 'pretrained' in config.model:
        config.model.pretrained = None
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """Deprecated.

    A simple pipeline to load image.
    """

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        warnings.simplefilter('once')
        warnings.warn('`LoadImage` is deprecated and will be removed in '
                      'future releases. You may use `LoadImageFromWebcam` '
                      'from `mmdet.datasets.pipelines.` instead.')
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


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
            data = dict(img_info=dict(filename=img), img_prefix=None)
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

"""
def inference_detector(model, imgs, batch_size, img_root=None, dim=256):
    Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    test_pipeline = Compose(cfg.test_pipeline)
    
    
    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)
        
    print(datas)
    
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
    print(results)
    
    
    
    imgs = Path(imgs)
    
    test_dataset = TestDataset(imgs, test_pipeline, img_root=img_root)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


    print('#### There are total {} imgs! ####'.format(len(test_dataset)))

    save_dic = {}
    input_size = cfg.img_scale
    with torch.no_grad():
        for d in tqdm(test_loader):
            print(d)
            d[0] = d[0].to(device)
            d[1] = d[1].to(device)
            result = model(d[0], d[1])
            print(result)
"""
            

def main(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)

    img = args.img
    #image_list = [os.path.join(img, image) for image in os.listdir(img)]
    result = inference_detector(model, img)
    # result = result.detach().cpu()
    #if args.out_path is None:
    #    out_path = Path(args.img)
    #    out_file = str(out_path.parent / '{}.pth'.format(out_path.name))
    #else:
        # img_path = Path(args.img)
    #    out_path = Path(args.out_path)
    #    out_path.mkdir(exist_ok=True, parents=True)
    #    out_file = str(out_path / '{}.pth'.format(args.depart))
    #torch.save(result, out_file)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, test_pipeline, img_root=None):
        # self.img_dir = img_dir
        if imgs.is_dir():
            imgs = imgs.glob('*.jpg')
            self.imgs = sorted([str(item) for item in imgs])
        elif imgs.is_file():
            imgs = json.load(open(imgs))['images']
            self.imgs = [os.path.join(img_root, img['file_name']) for img in imgs]
        self.test_pipeline = test_pipeline

    def __getitem__(self, idx):
        filename = self.imgs[idx]
        # filename = os.path.join(self.img_dir, filename)
        img_dict = dict(img_info=dict(filename=filename), img_prefix=None)
        data = self.test_pipeline(img_dict)
        #img = data['img'][0].data
        #img_metas = data['img_metas'][0].data
        return img #img, img_metas

    def __len__(self):
        """Total number of samples of data."""
        return len(self.imgs)

if __name__ == '__main__':
    args = parse_args()
    main(args)
