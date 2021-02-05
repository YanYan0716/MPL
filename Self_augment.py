'''
reference:
https://github.com/google-research/google-research/tree/1f1741a985a0f2e6264adae985bde664a7993bd2/flax_models/cifar/datasets
'''
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

'''
可能augment.py中的内容有问题 涉及文件augment.py的line 53，54
引用的库不一样，因为tensorflow.contrib已经停用，
使用的第三方：pip install tensorflow-addons
'''
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

import config
import augment
from self_aug_util import level_to_arg
from self_aug_func import *

# 将对图片做augment的函数变成一个字典
NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'Solarize': solarize_arg,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
    'Identity': identity,
}
# 在某些函数中有一些需要一个替换的值，比如旋转中有一些位置的像素值需要补充
REPLACE_FUNCS = frozenset({
    'Rotate',
    'TranslateX',
    'ShearX',
    'SHearY',
    'TranslateY',
    'Cutout',
})


class RandAugment(object):
    def __init__(self, num_layers=2, magnitude=None, cutout_const=40, translate_const=100., available_ops=None):
        '''
        reference: https://arxiv.org/abs/1909.13719
        :param num_layers:
        :param magnitude:
        :param cutout_const:
        :param translate_const:
        :param avalilable_ops:
        '''
        super(RandAugment, self).__init__()
        self.num_layers = num_layers
        self.cutout_const = float(cutout_const)
        self.translate_const = float(translate_const)
        if available_ops is None:
            available_ops = [
                'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
                'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout',
            ]
        self.available_ops = available_ops
        self.magnitude = magnitude

    def distort(self, image):
        '''

        :param image:  shape:[HWC] C=3
        :return: 返回一个经过变化后的图片
        '''
        input_image_type = image.dtype
        image = tf.clip_by_value(image, tf.cast(0, input_image_type), tf.cast(255, input_image_type))
        image = tf.cast(image, tf.uint8)

        replace_value = [128] * 3
        prob = tf.random.uniform([], 0.2, 0.8, tf.float32)

        for _ in range(self.num_layers):
            op_to_select = tf.random.uniform([], 0, int(len(self.available_ops)), dtype=tf.int32)
            branch_fns = []
            for (i, op_name) in enumerate(self.available_ops):
                func = NAME_TO_FUNC[op_name]  # 得到函数名称
                args = level_to_arg(self.cutout_const, self.translate_const)[op_name](self.magnitude)  # 得到func函数中所需要的参数

                # if op_name in REPLACE_FUNCS:
                #     args = lambda args, replace_value: tuple(list(args) + [replace_value])

                # 图像augment操作的实现
                def branch_fn(selected_func=func, selected_args=args, image=image):
                    if tf.random.uniform([], 0., 1., prob.dtype) <= prob:
                        image = selected_func(image, selected_args)
                    return image

                branch_fns.append((i, branch_fn))
            image = tf.switch_case(branch_index=op_to_select, branch_fns=branch_fns)
        image = tf.cast(image, dtype=input_image_type)
        return image


### ==============================


def unlabel_image(img_file, label):
    img = tf.io.read_file(img_file)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    ori_image = img  # 此图片作为原始图片

    aug = RandAugment(
        cutout_const=config.IMG_SIZE // 8,
        translate_const=config.IMG_SIZE // 8,
        magnitude=config.AUGMENT_MAGNITUDE,
    )

    aug_image = aug.distort(img)
    # aug_image = augment.cutout(aug_image, pad_size=config.IMG_SIZE // 8, replace=128)
    aug_image = tf.image.random_flip_left_right(aug_image)

    aug_image = tf.cast(aug_image, tf.float32) / 255.0
    ori_image = tf.cast(ori_image, tf.float32) / 255.0

    return {'ori_images': ori_image, 'aug_images': aug_image}


if __name__ == '__main__':
    df_unlabel = pd.read_csv('./dataset/unlabel/data.csv')
    file_paths = df_unlabel['file_name'].values
    labels = df_unlabel['label'].values

    ds_unlabel_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_unlabel_train = ds_unlabel_train.map(unlabel_image).batch(2)

    for data in ds_unlabel_train:
        aug_images = data['aug_images']
        ori_images = data['ori_images']
        plt.subplot(1, 2, 1)
        plt.imshow(ori_images[0].numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(aug_images[0].numpy())
        plt.show()
        break
