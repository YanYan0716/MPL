import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd

import config
import augment


def normalize_image(img, label):
    '''
    图片的归一化
    :param img:
    :param label:
    :return:
    '''
    return tf.cast(img, tf.float32)/255.0, label


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
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, (config.IMG_SIZE+5, config.IMG_SIZE+5))
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
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.image.random_flip_left_right(img)
    ori_image = img  # 此图片作为原始图片

    aug = augment.RandAugment(
        cutout_const=config.IMG_SIZE //8,
        translate_const=config.IMG_SIZE //8,
        magnitude=config.AUGMENT_MAGNITUDE,
    )
    aug_image = aug.distort(img)
    aug_image = augment.cutout(aug_image, pad_size=config.IMG_SIZE//4, replace=128)
    aug_image = tf.image.random_flip_left_right(aug_image)

    aug_image = tf.cast(aug_image, tf.float32)/ 255.0
    ori_image = tf.cast(ori_image, tf.float32)/ 255.0

    return {'ori_images': ori_image, 'aug_images': aug_image}


if __name__ == '__main__':
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # 有标签的数据集
    df_label = pd.read_csv(config.LABEL_FILE_PATH)
    file_paths = df_label['file_name'].values
    labels = df_label['label'].values
    ds_label_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_label_train = ds_label_train.map(label_image, num_parallel_calls=AUTOTUNE).batch(2)
        # .take(4000)\
        # .repeat()\
        # .shuffle(buffer_size=config.SHUFFLE_SIZE)\
        # .prefetch(buffer_size=AUTOTUNE)\
        # .batch(batch_size=config.BATCH_SIZE, drop_remainder=True)
    for data in ds_label_train:
        print(data.keys())
        break

    # 无标签的数据集
    df_unlabel = pd.read_csv(config.UNLABEL_FILE_PATH)
    file_paths = df_unlabel['file_name'].values
    labels = df_unlabel['label'].values
    ds_unlabel_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_unlabel_train = ds_unlabel_train.map(unlabel_image, num_parallel_calls=AUTOTUNE).batch(1)
        # .take(4000)\
        # .repeat()\
        # .shuffle(buffer_size=config.SHUFFLE_SIZE)\
        # .prefetch(buffer_size=AUTOTUNE)\
        # .batch(batch_size=config.BATCH_SIZE, drop_remainder=True)
    for data in ds_unlabel_train:
        print(type(data))
        # print(label)
        break


