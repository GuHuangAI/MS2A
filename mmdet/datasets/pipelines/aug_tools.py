#encoding=utf-8
from __future__ import print_function, division

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from scipy import ndimage, misc
import os
import cv2
import time
import copy


try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

np.random.seed(666)
ia.seed(666)

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
sometimes_3 = lambda aug: iaa.Sometimes(0.3, aug)
sometimes_1 = lambda aug: iaa.Sometimes(0.1, aug)
sometimes_01 = lambda aug: iaa.Sometimes(0.1, aug)
sometimes_05 = lambda aug: iaa.Sometimes(0.05, aug)
sometimes_02 = lambda aug: iaa.Sometimes(0.02, aug)
sometimes_03 = lambda aug: iaa.Sometimes(0.03, aug)
sometimes_7 = lambda aug: iaa.Sometimes(0.7, aug)
sometimes_8 = lambda aug: iaa.Sometimes(0.8, aug)

sometimes_10 = lambda aug: iaa.Sometimes(0.1, aug)
sometimes_20 = lambda aug: iaa.Sometimes(0.2, aug)
sometimes_30 = lambda aug: iaa.Sometimes(0.3, aug)
sometimes_40 = lambda aug: iaa.Sometimes(0.4, aug)
sometimes_50 = lambda aug: iaa.Sometimes(0.5, aug)
sometimes_60 = lambda aug: iaa.Sometimes(0.6, aug)
sometimes_70 = lambda aug: iaa.Sometimes(0.7, aug)
sometimes_80 = lambda aug: iaa.Sometimes(0.8, aug)
sometimes_90 = lambda aug: iaa.Sometimes(0.9, aug)



#加crop 的pad ，平移缩放
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        # iaa.Flipud(0.2),  # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes_3(iaa.Crop(percent=(0.05, 0.05))), # crop images from each side by 0 to 16px (randomly chosen)
        # sometimes_8(
        #     iaa.Pad(percent=(0.0001, 0.1),
        #         pad_mode=["constant", "maximum", "median", "minimum" ], #ia.ALL,  #####
        #         pad_cval=(0, 255))),
        # sometimes_8(iaa.CropAndPad(# crop 问题大 ，最好不要crop 太多， 要不外界判断一下问题。
        #     percent=(-0.0001, 0.3),
        #     pad_mode=["constant", "maximum", "median", "minimum" ], #ia.ALL,  #####
        #     pad_cval=(0, 255)
        # )),
        #  亮度调节
        sometimes_8(iaa.OneOf([
           iaa.Add((-10, 10), per_channel=0.5),
           iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
           iaa.Multiply((0.5, 1.5), per_channel=0.5),
           iaa.ContrastNormalization((0.5, 1.50), per_channel=0.5),  # improve or worsen the contrast
            ])),#sometimes(),

        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.5), "y": (0.8, 1.5)},
            # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            # translate by -20 to +20 percent (per axis)
            rotate=(-10, 10),  # rotate by -45 to +45 degrees  # 旋转过大 问题大
            shear=(-3, 3),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode= ia.ALL  #"constant"# use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # sometimes_3(iaa.Rot90([1,3])),
        sometimes_02(iaa.OneOf([
                            #iaa.Fog(),
                               iaa.Snowflakes(),
                               # iaa.Clouds(),
                               ]
                              )),
        # execute 0 to 3 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 2),
                   [
                       # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 2.5)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 5)),
                           # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 5)),
                           # blur image using local medians with kernel sizes between 2 and 7
                           # iaa.Lambda( # motion blur
                           #          func_images=func_images,
                           #          func_keypoints=func_keypoints
                           #      )
                           iaa.MotionBlur(k=[3,5]),
                       ]),
                       # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                       # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                       # search either for all edges or for directed edges,
                       # blend the result with the original image using a blobby mask
                       # iaa.SimplexNoiseAlpha(iaa.OneOf([
                       #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                       #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                       # ])),
                       # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images

                       iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                       # iaa.Snowflakes(),

                       # iaa.OneOf([
                       #     iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                       #     iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       # ]),
                       # iaa.Invert(0.05, per_channel=True), # invert color channels
                       # iaa.Add((-10, 10), per_channel=0.5),
                       # # change brightness of images (by -10 to 10 of original value)
                       # iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                       # # either change the brightness of the whole image (sometimes
                       # # per channel) or change the brightness of subareas
                       # iaa.Multiply((0.5, 1.5), per_channel=0.5),
                       # iaa.ContrastNormalization((0.5, 1.50), per_channel=0.5),  # improve or worsen the contrast

                       # iaa.OneOf([
                       #     iaa.Multiply((0.5, 1.5), per_channel=0.5),
                       #     iaa.FrequencyNoiseAlpha(
                       #         exponent=(-4, 0),
                       #         first=iaa.Multiply((0.5, 1.5), per_channel=True),
                       #         second=iaa.ContrastNormalization((0.5, 2.0))
                       #     )
                       # ]),
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       sometimes(iaa.ElasticTransformation(alpha=(0.01, 0.1), sigma=0.1)),## box不变的。
                       # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),  # sometimes move parts of the image around
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.03)))
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)





#加crop 的pad ，平移缩放
seq_croppaste = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        sometimes_8(iaa.CropAndPad(# crop 问题大 ，最好不要crop 太多， 要不外界判断一下问题。
            percent=(-0.05, 0.05),
            pad_mode=["constant",  "median"], #ia.ALL,  #####
            pad_cval=(0, 255)
        )),

        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.5), "y": (0.8, 1.5)},
            # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            # translate by -20 to +20 percent (per axis)
            rotate=(-2, 2),  # rotate by -45 to +45 degrees  # 旋转过大 问题大
            shear=(-2, 2),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode='constant' #"constant"# use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        #  亮度调节
        sometimes_8(iaa.OneOf([
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
            iaa.Multiply((0.5, 1.5), per_channel=0.5),
            iaa.ContrastNormalization((0.5, 1.50), per_channel=0.5),  # improve or worsen the contrast
        ])),  # sometimes(),

        iaa.SomeOf((0, 2), # 模糊变化
                   [
                       # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 2.5)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 5)),
                           # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 5)),
                           # blur image using local medians with kernel sizes between 2 and 7
                           # iaa.Lambda( # motion blur
                           #          func_images=func_images,
                           #          func_keypoints=func_keypoints
                           #      )
                           iaa.MotionBlur(k=[3,5]),
                       ]),

                       iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                       iaa.Snowflakes(),
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       sometimes(iaa.ElasticTransformation(alpha=(0.01, 0.1), sigma=0.1)),## box不变的。
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.03)))
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)


#不用平移缩放
seq_nocrop = iaa.Sequential(
    [
        sometimes_10(iaa.OneOf([
                            #iaa.Fog(),
                              iaa.Snowflakes(),
                              # iaa.Clouds(),
                              ]
                    )),
        #  亮度调节
        sometimes_50(iaa.OneOf([
           iaa.Add((-20, 20), per_channel=0.5),
           iaa.AddToHueAndSaturation((-10, 20)),  # change hue and saturation
           iaa.Multiply((0.5, 1.5), per_channel=0.5),
           iaa.ContrastNormalization((0.5, 1.50), per_channel=0.5),  # improve or worsen the contrast
           iaa.Grayscale(alpha=(0.0, 1.0)),
        ])),

        sometimes_20(
            iaa.OneOf([
                   iaa.GaussianBlur((0, 2.5)),  # blur images with a sigma between 0 and 3.0
                   iaa.AverageBlur(k=(2, 5)),
                   iaa.MedianBlur(k=(3, 5)),
                   iaa.MotionBlur(k=[3,5]),

               ])
        ),
        sometimes_20(
            iaa.OneOf([
                iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                sometimes(iaa.ElasticTransformation(alpha=(0.01, 0.1), sigma=0.1)),  ## box不变的。
                iaa.Cutout(nb_iterations=(1, 5), size=0.1, squared=False),
                iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))
            ])
        ),

    ],
    random_order=True
)



#不用平移缩放
seq_patch = iaa.Sequential(
    [
        # iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        sometimes_01(iaa.OneOf([
                              iaa.Snowflakes(),
                              ]
                    )),
        #  亮度调节
        sometimes_8(iaa.SomeOf((0, 4),([
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
            iaa.Multiply((0.5, 1.5), per_channel=0.5),
            iaa.ContrastNormalization((0.5, 1.50), per_channel=0.5),  # improve or worsen the contrast
            iaa.LinearContrast((0.21, 1.25), per_channel=0.5),  # improve or worsen the contrast
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03 * 255), per_channel=0.5),  # add gaussian noise to images
        ]))),

        sometimes(iaa.Affine(
            scale={"x": (0.99, 1.01), "y": (0.99, 1.01)},
            translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
            # translate by -20 to +20 percent (per axis)
            rotate=(-0, 0),  # rotate by -45 to +45 degrees  # 旋转过大 问题大
            shear=(-0, 0),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode="constant"
            # ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        iaa.SomeOf((0, 2),
                   [
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 2.5)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 5)),
                           iaa.MedianBlur(k=(3, 5)),
                           iaa.MotionBlur(k=[3,5]),
                       ]),
                       # iaa.Snowflakes(),
                       iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       sometimes(iaa.ElasticTransformation(alpha=(0.01, 0.1), sigma=0.1)),  ## box不变的。
                       # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),  # sometimes move parts of the image around
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.03)))
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)

seq_type_dict = {
    'seq_nocrop':seq_nocrop,
    'seq':seq,
    'seq_croppaste':seq_croppaste,

}

def judge_box(aug_det,bbs_augs,image):

    # 出边框的不要
    bbs_aug = bbs_augs[0]  # just one
    old_ln = len(bbs_aug.bounding_boxes)
    bbs_aug = bbs_aug.remove_out_of_image(fully=True, partly=False)  # remove bad box
    box_len = len(bbs_aug.bounding_boxes)
    if old_ln != box_len:
        pass
    #  print('remove bad box ',old_ln,len(bbs_aug.bounding_boxes))

    result_box = np.zeros((box_len, 4), dtype=np.float32)
    result_label = np.zeros(box_len, dtype=np.int32)
    tag = False
    for i, b in enumerate(bbs_aug.bounding_boxes):
        result_box[i, :] = [b.x1, b.y1, b.x2, b.y2]
        result_label[i] = b.label
        if (b.x1 > image.shape[1] or b.x2 > image.shape[1] or b.x1 < 0 or b.x2 < 0
                or b.y1 > image.shape[0] or b.y2 > image.shape[0] or b.y1 < 0 or b.y2 < 0):
            tag = True
            # 可不可能  crop 的box 不在这里面。
            break
    return tag ,result_box , result_label

def processimage(image,boxes,labels=None,type='seq_nocrop'):

    tpbb = []
    if labels is None:
        for box in boxes:
            tpbb.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
    else:
        for box,label in zip(boxes,labels):
            tpbb.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3],label=label))
    bbs = ia.BoundingBoxesOnImage(tpbb, shape=image.shape)
    augcls = seq_type_dict[type]
    # 加入边缘box检测
    h,w,c = image.shape
    out_tag = False
    th = 0.005
    for box in boxes:
        if box[0]/w < th or  box[2]/w > 1-th or box[1]/h < th or box[3]/h > 1-th:
            out_tag = True
            break
    if out_tag:
        augcls = seq_type_dict['seq_nocrop'] # 靠边的图，不croppad


    for testtime in range(10): # 强力
        aug_det = augcls.to_deterministic()
        bbs_augs = aug_det.augment_bounding_boxes([bbs])
        tag ,result_box , result_label = judge_box(aug_det,bbs_augs,image)
        if(not tag):
            image_aug = aug_det.augment_images([image])[0]
            if labels is not None :
                return image_aug, result_box,result_label
            return image_aug,result_box

    # 简单aug
    augcls = seq_type_dict['seq_nocrop']  # 靠边的图，不croppad
    for testtime in range(10): # easy
        aug_det = augcls.to_deterministic()
        bbs_augs = aug_det.augment_bounding_boxes([bbs])
        tag ,result_box , result_label = judge_box(aug_det,bbs_augs,image)
        if(not tag):
            image_aug = aug_det.augment_images([image])[0]
            if labels is not None :
                return image_aug, result_box,result_label
            return image_aug,result_box

    print('no augmetation ------',image.shape)
    if labels is not None :
        return image,boxes,labels
    return image,boxes


def processimage_test(image,boxes_min):
    # print('######',image.shape, image.dtype,image[0][0].dtype)
    # print (boxes.shape)
    boxes= boxes_min*[image.shape[0],image.shape[1],image.shape[0],image.shape[1]]
    print(boxes)

    tpbb = []
    for box in boxes:
        tpbb.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
    bbs = ia.BoundingBoxesOnImage(tpbb, shape=image.shape)

    aug_det = seq.to_deterministic()
    image_aug = aug_det.augment_images([image])[0]
    bbs_aug = aug_det.augment_bounding_boxes([bbs])[0]
    #
    result_box = np.zeros((len(boxes), 4), dtype=np.float32) #这是个天坑， 默认float 是64
    #
    for i,b in enumerate(bbs_aug.bounding_boxes):
        # boxes
        # result_box.append( np.array( [b.x1 , b.y1 , b.x2  , b.y2 ]).astype(np.int).astype(np.float)  )
        result_box[i,:] =np.array(  [b.x1 , b.y1 , b.x2  , b.y2 ] )/[image.shape[0],image.shape[1],image.shape[0],image.shape[1]]
    # print('------',image.shape)
    # result_box = result_box.astype(np.float32)
    print('-------------',result_box,result_box.dtype,image_aug.dtype)
    return image_aug,result_box


def readimageinfo(image_info,show=False):
    image = cv2.imread(image_info['filepath'])
    print(image_info['filepath'])
    tpbb = []
    for bbox in image_info['bboxes']:
        x1 = bbox['x1']
        x2 = bbox['x2']
        y1 = bbox['y1']
        y2 = bbox['y2']
        if show: cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 12)
        tpbb.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
    bbs = ia.BoundingBoxesOnImage(tpbb, shape=image.shape)
    if show:
        cv2.imshow('origin',image)
        cv2.waitKey()
    return image,bbs





#test
seq_crop = iaa.Sequential(
    [ iaa.MotionBlur(k=[3,5]),],
    random_order=True)



def process_aug_tools_test(image_info):
    image, bbs = readimageinfo(image_info)
    import time
    start = time.clock()
    test = []
    for i in range(100):
        tag=True
        while tag:
            tag = False
            aug_det = seq_crop.to_deterministic()
            image_aug = aug_det.augment_images([image])[0]
            bbs_aug = aug_det.augment_bounding_boxes([bbs])[0]
            bbs_aug = bbs_aug.remove_out_of_image().cut_out_of_image()
            image_after = bbs_aug.draw_on_image(image_aug, thickness=10, color=[255, 0, 0])
            result_box = np.zeros((len(bbs.bounding_boxes), 4), dtype=np.float32)
            for ii, b in enumerate(bbs_aug.bounding_boxes):
                # boxes
                # result_box.append( np.array( [b.x1 , b.y1 , b.x2  , b.y2 ]).astype(np.int).astype(np.float)  )
                # result_box[i,:] = [b.x1 , b.y1 , b.x2  , b.y2 ]
                print('---:\n',[b.x1, b.y1, b.x2, b.y2])
                result_box[ii, :] = np.array([b.x1, b.y1, b.x2, b.y2]) / [image.shape[1], image.shape[0], image.shape[1],
                                                                         image.shape[0]]
                print (result_box[ii, :])
                if( np.any( result_box[ii] >1) or np.any(result_box[ii] <0.)):
                    tag=True

        # # 第二步
        #
        # tag=True
        # while tag:
        #     tag = False
        #     bbs = ia.BoundingBoxesOnImage(bbs_aug.bounding_boxes, shape=image_after.shape)
        #     aug_det = seq_crop.to_deterministic()
        #     image_aug = aug_det.augment_images([image_after])[0]
        #     bbs_aug = aug_det.augment_bounding_boxes([bbs])[0]
        #     image_after = bbs_aug.draw_on_image(image_after, thickness=10, color=[0, 255, 0])
        #     result_box = np.zeros((len(bbs.bounding_boxes), 4), dtype=np.float32)
        #     for ii, b in enumerate(bbs_aug.bounding_boxes):
        #         # boxes
        #         # result_box.append( np.array( [b.x1 , b.y1 , b.x2  , b.y2 ]).astype(np.int).astype(np.float)  )
        #         # result_box[i,:] = [b.x1 , b.y1 , b.x2  , b.y2 ]
        #         print('---:\n',[b.x1, b.y1, b.x2, b.y2])
        #         result_box[ii, :] = np.array([b.x1, b.y1, b.x2, b.y2]) / [image_after.shape[1], image_after.shape[0], image_after.shape[1],
        #                                                                   image_after.shape[0]]
        #         print (result_box[ii, :])
        #         if( np.any( result_box[ii] >1) or np.any(result_box[ii] <0.)):
        #             tag=True

        test.append(image_aug )
        # grid = seq.draw_grid(image_aug, cols=8, rows=8)
        basename = str(i)+os.path.basename(image_info['filepath'])
        cv2.imwrite('./'+basename, image_after)
        print('cv write ok ')
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)


#  mask 支持
OUTPUTDIR = './testimageaug/'

def process_mask_poin_aug(image,boxs,segms,points,imagepath=None, show=True):
    # box xyxy   N * 4
    # segms 【[1,2,3,4],】  N * m  一个mask 可能有多个seg区域 .
    #  points N * 3 * 21
    out_image_tag = False
    pad = int(image.shape[0]*0.1) # 框边界5个像素的就不要移动了
    tpbb = []
    if  show:
        showimage = copy.deepcopy(image)
        if not os.path.exists(OUTPUTDIR):
            os.makedirs(OUTPUTDIR)
    for bbox in boxs:
        # x1 = bbox[0]
        # x2 = bbox[2]
        # y1 = bbox[1]
        # y2 = bbox[3]
        x1,y1,x2,y2 = bbox
        if np.any(np.array(bbox) < pad):
            out_image_tag = True
        if np.any(np.array(bbox[1::2]) > image.shape[1]-pad) or np.any(np.array(bbox[0::2]) > image.shape[0] - pad):
            out_image_tag = True
        if show:
            cv2.rectangle(showimage, (x1, y1), (x2, y2), (0, 255, 0), 12)
            #cv2.drawContours(showimage, contours, -1, (0, 0, 255), 10)
            # cv2.imshow('image', showimage)

        tpbb.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
    bbs = ia.BoundingBoxesOnImage(tpbb, shape=image.shape)
    if show:
        basename = os.path.basename(imagepath)
        cv2.imwrite(os.path.join(OUTPUTDIR ,'input_' + basename), showimage)
        # if segms is not None:
        #     image_segpoint_input = copy.deepcopy(image)
        #     for seg_ in segms: # 一个分割手区域 有多个分割组合
        #         for seg in seg_:
        #             for idx in range(len(seg)//2):
        #                 cv2.circle(image_segpoint_input, (int(seg[idx*2]),int(seg[idx*2+1])),5 , (0,255,0),3)
        #     cv2.imwrite('./testimage/' + 'input_seg_' + basename, image_segpoint_input)
        # if points is not None:
        #     image_keypoint_input = copy.deepcopy(image)
        #     for point in points:
        #         for idx in range(len(point[0])) :
        #             cv2.circle(image_keypoint_input, (int(point[0,idx]), int(point[1,idx ])), 2, (0, 0, 255), 3)
        #     cv2.imwrite('./testimage/' + 'input_key_' + basename, image_keypoint_input)

    if out_image_tag:
        using_seq = seq_nocrop
    else:
        using_seq = seq

    for testtime in range(6):
        aug_det = using_seq.to_deterministic()
        try:
            bbs_aug = aug_det.augment_bounding_boxes([bbs])[0]
        except Exception:
            import traceback
            traceback.print_exc()
            print('imagepath @@@ ', imagepath)
            # print('imagepath @@@ ',imagepath,boxs,segms,points,)
            # print('bbs',bbs)
            continue

        result_box = np.zeros((len(bbs.bounding_boxes), 4), dtype=np.float32)
        tag = False
        for i, b in enumerate(bbs_aug.bounding_boxes):
            # 加入旋转后， box 有扩大的问题
            result_box[i, :] = [b.x1, b.y1, b.x2, b.y2]
            if (b.x1 >= image.shape[1] or b.x2 >= image.shape[1] or b.x1 <= 0 or b.x2 <= 0
                    or b.y1 >= image.shape[0] or b.y2 >= image.shape[0] or b.y1 <= 0 or b.y2 <= 0):
                tag = True
                break
        if not tag:
            image_aug = aug_det.augment_images([image])[0]
            if show:
                print('old bbox',boxs)
                tempbox = copy.deepcopy(boxs)
            for i, bbox in enumerate(boxs):
                # 并不会修改数据
                bbox = [bbs_aug.bounding_boxes[i].x1,bbs_aug.bounding_boxes[i].y1,bbs_aug.bounding_boxes[i].x2,bbs_aug.bounding_boxes[i].y2]
            #     # tpbb.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
            if show:
                print('diff bbox', np.array(tempbox) - np.array(result_box), np.array(tempbox) - np.array(boxs),)
                print('new bbox',   np.array(result_box))

            if segms is not None:
                if show:
                    print('old segms', segms)
                    print(' shift box ', result_box)
                keys_segs = []
                for segs in segms:
                    for seg in segs:
                        for i in range(int(len(seg) / 2)):
                            keys_segs.append(ia.Keypoint(x=int(seg[i * 2]), y=int(seg[i * 2 + 1])))
                keys_segments = ia.KeypointsOnImage(keys_segs, shape=image.shape)
                segpoint = aug_det.augment_keypoints([keys_segments])[0]

                segs_index = 0
                for ind,segs in enumerate(segms):
                    x1, y1 = image.shape[1] - 1, image.shape[0] -1
                    x2, y2 = 0, 0
                    for seg in segs:
                        for i in range(int(len(seg) / 2)):
                            seg[i * 2] = segpoint.keypoints[segs_index].x
                            seg[i * 2 + 1] = segpoint.keypoints[segs_index].y
                            segs_index += 1
                            # keys_segs.append(ia.Keypoint(x=int(seg[i * 2]), y=int(seg[i * 2 + 1])))
                        segx1 = min(seg[::2])
                        segy1 = min(seg[1::2])
                        segx2 = max(seg[::2])
                        segy2 = max(seg[1::2])
                        if(segx1 < x1):
                            x1 = segx1
                        if(segy1 < y1):
                            y1 = segy1
                        if(segx2 > x2):
                            x2 = segx2
                        if(segy2 > y2):
                            y2 = segy2
                    # 用分割支路 精细 box
                    # 加入旋转后， box 有扩大的问题
                    if show:
                        print(' segfixed    ', [x1, y1, x2, y2])
                    if ( x1<x2 and y1<y2  ):     # 可能有空的seg
                        result_box[ind, :] = np.array([x1, y1, x2, y2]).astype(np.float32)
                if show:
                    print(' segfixed  box ', result_box)
                    print('new segms', segms)
            if points is not None:
                keys_point = []
                for p in points:
                    for i in range(p.shape[1]):
                        keys_point.append(ia.Keypoint(x=int(p[0, i]), y=int(p[1, i])))
                keys_points = ia.KeypointsOnImage(keys_point, shape=image.shape)
                keypoint = aug_det.augment_keypoints([keys_points])[0]

                if show:
                    print('old point', points)
                    temp_points = copy.deepcopy(points)
                points_index = 0
                for p in points:
                    for i in range(p.shape[1]):
                        p[0, i] = keypoint.keypoints[points_index].x
                        p[1, i] = keypoint.keypoints[points_index].y
                        points_index += 1
                        # keys_point.append(ia.Keypoint(x=int(p[0, i]), y=int(p[1, i])))
                if show:
                    print('diff point', np.array(temp_points) - np.array(points) )

            if show:
                #################todo ####################
                # 这里全部换自己的绘图。
                basename = os.path.basename(imagepath)
                # image_after = bbs_aug.draw_on_image(copy.deepcopy(image_aug), thickness=2, color=[255, 0, 0])
                # # cv2.imshow('image_after',image_after)
                # cv2.imwrite('./testimage/' + 'box_' + basename, image_after)
                boximage = copy.deepcopy(image_aug).copy()
                for bo in result_box:
                    cv2.rectangle(boximage, (bo[0] , bo[1] ), (bo[2], bo[3] ), (255, 0, 0), 2)
                cv2.imwrite(os.path.join(OUTPUTDIR, 'box_hand_' + basename), boximage)
                # if segms is not None:
                #     image_segpoint = segpoint.draw_on_image(image_aug, color=[0, 255, 255], size=3, copy=True)
                #     cv2.imwrite('./testimage/' + 'seg_' + basename, image_segpoint)
                # # cv2.imshow('image_segpoint', image_segpoint)
                # if points is not None:
                #     image_keypoint = keypoint.draw_on_image(image_aug, color=[0, 255, 0], size=3, copy=True)
                #     cv2.imwrite('./testimage/' + 'key_' + basename, image_keypoint)
                # cv2.imshow('image_keypoint', image_keypoint)
                # cv2.waitKey(0)
                image = image_aug  #图没改啊
            return image_aug,result_box,segms,points # 瞎了，尽然写错变量了！！!

    print('no augmetation ------',image.shape,imagepath)
    return image,boxs,segms,points



#### patch 的aug 设置 ，
def process_patch_aug(image,boxs,segms,points,imagepath=None, show=True):
    aug_det = seq_patch.to_deterministic()  # 反正不移动box 直接出来了
    image_aug = aug_det.augment_images([image])[0]
    return image_aug, boxs, None, None



def single_mask_2ploy(mask):
    #xyxy
    Y, X = np.where(mask >= 1)
    return max(0, min(X)), max(0, min(Y)), min(mask.shape[1], max(X)), min(mask.shape[0], max(Y))

def process_singlemask(mask):
    mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
     # 存在一些有 分开的 部分手。 或者瑕疵。  要膨胀一下？？
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            segmentation.append(contour)
        # if len(contour) > 4: # 有只有两点的。这种就不要了 再json.python 里面会过滤小于6个点的。
        #     segmentation.append(contour)
    if len(segmentation) == 0:
        print('no contour')
        return None
    return segmentation,contours

def test_mask_and_point():
    path_to_db = 'E:\\RHD_v1-1\\RHD_published_v2'
    setname = 'evaluation'
    import pickle
    with open(os.path.join(path_to_db, setname, 'anno_%s.pickle' % setname), 'rb') as fi:
        anno_all = pickle.load(fi)
    print(type(anno_all.items()))
    sample_id, anno = list(anno_all.items())[5]
    sample_id, anno = list(anno_all.items())[188]
    dirpath = os.path.join(path_to_db, setname)

    for i in range(100, 200, 5):
        sample_id, anno = list(anno_all.items())[i]
        process_aug_tools_mask_point_test(anno, sample_id, dirpath)


def process_aug_tools_mask_point_test(anno,sample_id,dirpath , show=False):
    import os
    import scipy.misc
    imagepath = os.path.join(dirpath, 'color', '%.5d.png' % sample_id)
    image = scipy.misc.imread(imagepath)
    mask = scipy.misc.imread(os.path.join(dirpath, 'mask', '%.5d.png' % sample_id))

    lefthand_mask = (mask >= 2) & (mask <= 17)
    right_hand_mask = (mask >= 18)
    lefthand_point = anno['uv_vis'][:21]
    righthand_point = anno['uv_vis'][21:]

    if np.sum(lefthand_mask) > 10:
        mask_single = lefthand_mask
    else:
        mask_single = right_hand_mask

    hand_box = single_mask_2ploy(mask_single)
    bbox = [hand_box[0], hand_box[1], (hand_box[2] - hand_box[0]), (hand_box[3] - hand_box[1])]
    segment,contours = process_singlemask(mask_single)

    keys_segs = []
    for i in range(int(len(segment[0])/2)):
        keys_segs.append(ia.Keypoint(x=int(segment[0][i*2]), y=int(segment[0][i*2+1])))

    keys_segments = ia.KeypointsOnImage(keys_segs, shape=image.shape)

    keys_point = []
    for p in anno['uv_vis']:
        keys_point.append(ia.Keypoint(x=int(p[0]), y=int(p[1])))

    keys_points = ia.KeypointsOnImage(keys_point, shape=image.shape)

    image = cv2.imread(imagepath)
    tpbb = []
    x1 = hand_box[0]
    x2 = hand_box[2]
    y1 = hand_box[1]
    y2 = hand_box[3]

    showimage = copy.deepcopy(image)
    if show:
        cv2.rectangle(showimage, (x1, y1), (x2, y2), (0, 255, 0), 12)
        cv2.drawContours(showimage, contours, -1, (0, 0, 255), 10)
        cv2.imshow('image', showimage)
    tpbb.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
    bbs = ia.BoundingBoxesOnImage(tpbb, shape=image.shape)

    start = time.clock()
    test = []
    for i in range(5):
        tag=True
        while tag:
            tag = False
            seq_crop = seq
            seq_crop = seq_nocrop
            aug_det = seq_crop.to_deterministic()
            image_aug = aug_det.augment_images([image])[0]
            bbs_aug = aug_det.augment_bounding_boxes([bbs])[0]
            segpoint = aug_det.augment_keypoints([keys_segments])[0]
            keypoint = aug_det.augment_keypoints([keys_points])[0]
            image_after = bbs_aug.draw_on_image(copy.deepcopy(image_aug), thickness=5, color=[255, 0, 0])

            if show:cv2.imshow('image_after',image_after)
            image_segpoint = segpoint.draw_on_image(image_aug, color=[0, 255, 255], size=3, copy=True)
            if show:cv2.imshow('image_segpoint', image_segpoint)

            image_keypoint = keypoint.draw_on_image(image_aug, color=[0, 255, 0], size=3, copy=True)
            if show:cv2.imshow('image_keypoint', image_keypoint)
            if show:cv2.waitKey(0)



            result_box = np.zeros((len(bbs.bounding_boxes), 4), dtype=np.float32)
            for ii, b in enumerate(bbs_aug.bounding_boxes):
                # boxes
                # # result_box.append( np.array( [b.x1 , b.y1 , b.x2  , b.y2 ]).astype(np.int).astype(np.float)  )
                # # result_box[i,:] = [b.x1 , b.y1 , b.x2  , b.y2 ]
                # print('---:\n',[b.x1, b.y1, b.x2, b.y2])
                # result_box[ii, :] = np.array([b.x1, b.y1, b.x2, b.y2]) / [image.shape[1], image.shape[0], image.shape[1],
                #                                                          image.shape[0]]
                # print (result_box[ii, :])
                # if( np.any( result_box[ii] >1) or np.any(result_box[ii] <0.)):
                #     tag=True

                result_box[ii, :] = [b.x1, b.y1, b.x2, b.y2]
                if (b.x1 > image.shape[1] or b.x2 > image.shape[1] or b.x1 <= 0 or b.x2 <= 0
                        or b.y1 > image.shape[0] or b.y2 > image.shape[0] or b.y1 <= 0 or b.y2 <= 0):
                    tag = True
                    break

            boximage = copy.deepcopy(image_aug).copy()
            for bo in result_box:
                cv2.rectangle(boximage, (bo[0] , bo[1] ), (bo[2], bo[3] ), (255, 0, 0), 2)

        test.append(image_aug)
        # grid = seq.draw_grid(image_aug, cols=8, rows=8)
        basename = str(i)+os.path.basename(imagepath)

        cv2.imwrite('./' + 'box_hands' + basename, boximage)


        cv2.imwrite('./' + 'box_' + basename, image_after)
        cv2.imwrite('./' + 'seg_' + basename, image_segpoint)
        cv2.imwrite('./' + 'key_' + basename, image_keypoint)
        print('cv write ok ')
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)

if __name__ == "__main__":
    # main()
    test_mask_and_point()

    # image_info ={}
    # image_info['filepath']='E:\\data\\hand3\\data5\\houjiaying\\single_class\\train\\hand_3\\images\\9d9ab1ea33a62d1fbf5888aa0bcbb7fa_0013.png'
    # image_info['bboxes'] = [ {'x1':59,'y1':379,'x2':244,'y2':687} ,{'x1':276,'y1':375,'x2':496,'y2':705}]
    #
    # image_info[
    #     'filepath'] = 'E:\\data\\hand3\\data5\\houjiaying\\single_class\\train\\hand_3\\images\\9d9ab1ea33a62d1fbf5888aa0bcbb7fa_0041.png'
    # image_info['bboxes'] = [{'x1': 1, 'y1': 399, 'x2': 210, 'y2': 841}, {'x1': 180, 'y1': 514, 'x2': 398, 'y2': 911}]
    #
    #
    #
    # image_info[
    #     'filepath'] = 'E:\\data\\hand3\\data5\\houjiaying\\single_class\\train\\hand_3\\images\\9d9ab1ea33a62d1fbf5888aa0bcbb7fa_0050.png'
    # image_info['bboxes'] = [{'x1': 278, 'y1': 525, 'x2': 503, 'y2': 881}, {'x1': 1, 'y1': 574, 'x2': 279, 'y2': 960}]
    #
    # image_info[
    #     'filepath'] = 'E:\\data\\hand3\\data5\\houjiaying\\single_class\\train\\hand_3\\images\\9d9ab1ea33a62d1fbf5888aa0bcbb7fa_0065.png'
    # image_info['bboxes'] = [{'x1': 58, 'y1': 412, 'x2': 308, 'y2': 745}, {'x1': 367, 'y1': 355, 'x2': 544, 'y2': 960}]
    #
    # process_aug_tools_test(image_info)
    #
    # image = cv2.imread(
    #     'E:\\data\\hand3\\data5\\houjiaying\\single_class\\train\\hand_3\\images\\9d9ab1ea33a62d1fbf5888aa0bcbb7fa_0041.png')
    # degree = 50  # 运动的长度
    # angle = 30  # 方向

    # import numpy as np
    #
    # image = np.array(image)
    # # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    # M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    # motion_blur_kernel = np.diag(np.ones(degree))
    # motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    #
    # motion_blur_kernel = motion_blur_kernel / degree
    # blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # # convert to uint8
    # cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    # blurred = np.array(blurred, dtype=np.uint8)
    #
    # cv2.imshow('ne2w', blurred)
    # cv2.waitKey()


    pass
