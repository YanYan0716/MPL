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


def normalize_image(img, label):
    '''
    图片的归一化
    :param img:
    :param label:
    :return:
    '''
    return tf.cast(img, tf.float32) / 255.0, label


# 制作有标签的数据集
def label_image(img_file, label):
    '''
    获取图片，对图片做水平翻转 随机剪裁等， label变为onehot
    :param img_file:
    :param label:
    :return:
    '''
    # 对图片的处理
    img = tf.io.read_file(img_file)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, (config.IMG_SIZE + 5, config.IMG_SIZE + 5))
    img = tf.image.random_crop(img, (config.IMG_SIZE, config.IMG_SIZE, 3))
    img = tf.cast(img, tf.float32) / 255.0
    # 对标签的处理
    label = tf.raw_ops.OneHot(indices=label, depth=config.NUM_CLASSES, on_value=1.0, off_value=0)
    return {'images': img, 'labels': label}


# 制作无标签的数据集
def unlabel_image(img_file, label):
    '''
    处理无标签数据
    :param img_file:
    :param label:
    :return: 两张图片，一张经过轻微变换后的图片称为ori_image 一张经过较为剧烈变化后的图片，称为aug_images
    '''
    img = tf.io.read_file(img_file)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    ori_image = img  # 此图片作为原始图片

    aug = augment.RandAugment(
        cutout_const=config.IMG_SIZE // 8,
        translate_const=config.IMG_SIZE // 8,
        magnitude=config.AUGMENT_MAGNITUDE,
    )
    aug_image = aug.distort(img)
    aug_image = augment.cutout(aug_image, pad_size=config.IMG_SIZE // 8, replace=128)
    aug_image = tf.image.random_flip_left_right(aug_image)

    aug_image = tf.cast(aug_image, tf.float32) / 255.0
    ori_image = tf.cast(ori_image, tf.float32) / 255.0

    return {'ori_images': ori_image, 'aug_images': aug_image}


def merge_dataset(label_data, unlabel_data):
    return label_data['images'], label_data['labels'], unlabel_data['ori_images'], unlabel_data['aug_images']


if __name__ == '__main__':

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # 有标签的数据集 batch_size=config.BATCH_SIZE
    df_label = pd.read_csv(config.LABEL_FILE_PATH)
    file_paths = df_label['file_name'].values
    labels = df_label['label'].values
    ds_label_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_label_train = ds_label_train\
        .map(label_image, num_parallel_calls=AUTOTUNE)\
        .batch(1).shuffle(1)
    for data in ds_label_train:
        # print(data.keys())
        break

    # 无标签的数据集 batch_size=config.BATCH_SIZE*config.UDA_DATA
    df_unlabel = pd.read_csv(config.UNLABEL_FILE_PATH)
    file_paths = df_unlabel['file_name'].values
    labels = df_unlabel['label'].values
    ds_unlabel_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_unlabel_train = ds_unlabel_train\
        .map(unlabel_image, num_parallel_calls=AUTOTUNE)\
        .batch(1*config.UDA_DATA).shuffle(buffer_size=3)

    for data in ds_unlabel_train:
        # plt.figure(figsize=(10, 10))
        aug_images = data['aug_images']
        ori_images = data['ori_images']
        # plt.subplot(1, 2, 1)
        # plt.imshow(aug_images[0].numpy())
        # plt.subplot(1, 2, 2)
        # plt.imshow(ori_images[0].numpy())
        # plt.show()
        break

    # 将有标签数据和无标签数据整合成最终的数据形式
    ds_train = tf.data.Dataset.zip((ds_label_train, ds_unlabel_train))
    ds_train = ds_train.map(merge_dataset)
    for data in ds_train:
        label_img = data[0]
        print(data[0].shape)
        print(data[1].shape)
        print(data[2].shape)
        print(data[3].shape)
        label = data[1]
        ori_images = data[2]
        aug_images = data[3]
        print(label)
        print(label.shape)
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(label_img[0].numpy())
        plt.subplot(1, 3, 2)
        plt.imshow(ori_images[0].numpy())
        plt.subplot(1, 3, 3)
        plt.imshow(aug_images[0].numpy())
        plt.show()
        break
