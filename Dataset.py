import tensorflow as tf
import pandas as pd

import config


def read_image(img_file, label):
    '''
    获取图片，对图片做水平翻转 随机剪裁
    :param img_file:
    :param label:
    :return:
    '''
    img = tf.io.read_file(img_file)
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_crop(img, (config.IMG_SIZE, config.IMG_SIZE))
    return img, label


def normalize_image(img, label):
    '''
    图片的归一化
    :param img:
    :param label:
    :return:
    '''
    return tf.cast(img, tf.float32)/255.0, label


if __name__ == '__main__':
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    df = pd.read_csv(config.FILE_PATH)
    file_paths = df['file_name'].values
    labels = df['label'].values

    ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_train = ds_train.map(read_image, num_parallel_calls=AUTOTUNE)\
        .map(normalize_image, num_parallel_calls=AUTOTUNE)\
        .take(4000)\
        .repeat()\
        .shuffle(buffer_size=config.SHUFFLE_SIZE)\
        .prefetch(buffer_size=AUTOTUNE)\
        .batch(batch_size=config.BATCH_SIZE, drop_remainder=True)



