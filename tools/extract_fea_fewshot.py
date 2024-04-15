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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--depart', help='depart', default='big_18')
    parser.add_argument('--img', help='img file dir or json', default='')
    parser.add_argument('--img_root', help='if the input of --img is a json file, this must not be none', default=None)
    parser.add_argument('--config', help='Config file',
                        default='./configs/yolox/configs_yolox_deploy_yolox_l_8x8_300e_small_big18baseline_resizeinput_newdata_300.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',
                        default='/nfs/datamonitor/deploymodel/yolox_l_8x8_300e_small_big18baseline_resizeinput_newdata_300/epoch_300.pth')

    parser.add_argument('--out_path', default='/home/ml/code/mmdetection/fewshot_fea', help='Path to output file')
    parser.add_argument('--batch_size', default=8, type=int, help='test batch size')
    parser.add_argument('--dim', default=256, type=int, help='feature dim')
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


def inference_detector(model, imgs, batch_size, img_root=None, dim=256):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    # cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    # pipeline = cfg.data.test['pipeline']
    # test_pipeline = Compose(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.test_pipeline)
    # batch_size = min(batch_size, len(imgs))

    # datas = []
    # for img in imgs:
    #     # prepare data
    #     if isinstance(img, np.ndarray):
    #         # directly add img
    #         data = dict(img=img)
    #     else:
    #         # add information into dict
    #         data = dict(img_info=dict(filename=img), img_prefix=None)
    #     # build the data pipeline
    #     data = test_pipeline(data)
    #     datas.append(data)
    imgs = Path(imgs)
    test_dataset = TestDataset(imgs, test_pipeline, img_root=img_root)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # data = collate(datas, samples_per_gpu=len(imgs))
    # data = collate(datas, samples_per_gpu=batch_size)
    # just get the actual data from DataContainer
    # data['img_metas'] = [img_metas.data for img_metas in data['img_metas']]
    # data['img'] = [img.data for img in data['img']]
    # print('#### Collecting data finished! ####')
    print('#### There are total {} imgs! ####'.format(len(test_dataset)))
    # if next(model.parameters()).is_cuda:
    #     # scatter to specified GPU
    #     data = scatter(data, [device])[0]
    # else:
    #     for m in model.modules():
    #         assert not isinstance(
    #             m, RoIPool
    #         ), 'CPU inference with RoIPool is not supported currently.'
    save_dic = {}
    input_size = cfg.img_scale
    h, w = input_size[0], input_size[1]
    fea_dim = h * w // 64 + h * w // 1024 + h * w // 256
    cls = torch.zeros(len(test_dataset), fea_dim)
    obj = torch.zeros(len(test_dataset), fea_dim)
    areas = torch.zeros(len(test_dataset), fea_dim)
    n = 0
    feats1 = torch.zeros(len(test_dataset), dim, h // 8, w // 8)
    feats2 = torch.zeros(len(test_dataset), dim, h // 16, w // 16)
    feats3 = torch.zeros(len(test_dataset), dim, h // 32, w // 32)
    # forward the model
    with torch.no_grad():
        # results = model(return_loss=False, rescale=True, **data)
        for d in tqdm(test_loader):
            d = d.to(device)
            bs = d.shape[0]
            feat = model.extract_feat(d)
            # cls_score, bbox_pred, objectness = model.bbox_head.forward(feat)
            # featmap_sizes = [cls.shape[2:] for cls in cls_score]
            # mlvl_priors = model.bbox_head.prior_generator.grid_priors(
            #     featmap_sizes,
            #     dtype=cls_score[0].dtype,
            #     device=cls_score[0].device,
            #     with_stride=True)
            # bbox_pred = torch.cat([
            #     bbox.permute(0, 2, 3, 1).reshape(bs, -1, 4)
            #     for bbox in bbox_pred
            # ], dim=1)
            # mlvl_priors = torch.cat(mlvl_priors)
            # bbox_pred = model.bbox_head._bbox_decode(mlvl_priors, bbox_pred)
            # area = (bbox_pred[:, :, 2] - bbox_pred[:, :, 0]) * (bbox_pred[:, :, 3] - bbox_pred[:, :, 1])
            # cls_score = torch.cat([
            #     cls_pred.permute(0, 2, 3, 1).reshape(bs, -1)
            #     for cls_pred in cls_score
            # ], dim=1)
            # objectness = torch.cat([
            #     obj.permute(0, 2, 3, 1).reshape(bs, -1)
            #     for obj in objectness
            # ], dim=1)
            #print(cls.shape)
            # cls[n: n+bs] = cls_score  # without sigmoid
            # obj[n: n+bs] = objectness # without sigmoid
            # areas[n: n+bs] = area
            # score = cls_score.sigmoid() * objectness.sigmoid()
            # score = score * areas
            # save_tensor[n: n+bs] = score
            feats1[n: n+bs] = feat[0]
            feats2[n: n + bs] = feat[1]
            feats3[n: n + bs] = feat[2]
            n += bs
        save_dic['feats1'] = feats1.detach().cpu()
        save_dic['feats2'] = feats2.detach().cpu()
        save_dic['feats3'] = feats3.detach().cpu()
    return save_dic
    # if not is_batch:
    #     return results[0]
    # else:
    #     return results

def main(args):
    # build the model from a config file and a checkpoint file

    model = init_detector(args.config, args.checkpoint, device=args.device)

    img = args.img
    result = inference_detector(model, img, batch_size=args.batch_size, img_root=args.img_root, dim=args.dim)
    # result = result.detach().cpu()
    if args.out_path is None:
        out_path = Path(args.img)
        out_file = str(out_path.parent / '{}.pth'.format(out_path.name))
    else:
        # img_path = Path(args.img)
        out_path = Path(args.out_path)
        out_path.mkdir(exist_ok=True, parents=True)
        out_file = str(out_path / '{}.pth'.format(args.depart))
    torch.save(result, out_file)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, test_pipeline, img_root=None):
        # self.img_dir = img_dir
        if imgs.is_dir():
            imgs = imgs.glob('*.png')
            self.imgs = sorted([str(item) for item in imgs])
        elif imgs.is_file():
            imgs = json.load(open(imgs))['images']
            self.imgs = [os.path.join(img_root, img['file_name']) for img in imgs]
            print(len(self.imgs))
        self.test_pipeline = test_pipeline

    def __getitem__(self, idx):
        filename = self.imgs[idx]
        # filename = os.path.join(self.img_dir, filename)
        img_dict = dict(img_info=dict(filename=filename), img_prefix=None)
        data = self.test_pipeline(img_dict)
        img = data['img'][0].data
        return img

    def __len__(self):
        """Total number of samples of data."""
        return len(self.imgs)

if __name__ == '__main__':
    args = parse_args()
    main(args)
