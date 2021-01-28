import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


import config
from Model import Wrn28k


if __name__ == '__main__':
    # 制作数据
    batch_size = 1 # config.BATCH_SIZE
    uda_data = int(config.UDA_DATA)

    l_images = np.random.random((1, 32, 32, 3))
    l_images = tf.convert_to_tensor(l_images, dtype=tf.float32)
    aug_images = np.random.random((1 * config.UDA_DATA, 32, 32, 3))
    aug_images = tf.convert_to_tensor(aug_images, dtype=tf.float32)
    u_aug_and_l_images = tf.concat([aug_images, l_images], axis=0)

    # 构建student模型，产生输出
    student = Wrn28k(num_inp_filters=3, k=2)
    logits = {}
    cross_entroy = {}

    # 第一次call student -----------------------------
    logits['s_on_aug_and_l'] = student(x=u_aug_and_l_images)  # shape=[8, 10]
    logits['s_on_u'], logits['s_on_l_old'] = tf.split(
        logits['s_on_aug_and_l'],
        [aug_images.shape[0], l_images.shape[0]],
        axis=0
    )
    # print(logits['s_on_u'].shape, logits['s_on_l_old'].shape)
    cross_entroy['s_on_u'] = tf.losses.CategoricalCrossentropy(
        label_smoothing=config.LABEL_SMOOTHING,
        logits=True,
        reduction=tf.keras.losses.Reduction.NONE,
    )
    cross_entroy['s_on_u'] = tf.reduce_sum(cross_entroy['s_on_u']) / \
                             tf.convert_to_tensor(batch_size*uda_data, dtype=tf.float32)

    # for taylor
    cross_entroy['s_on_l_old'] = tf.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM,
    )
    shadow = tf.Variable(initial_value=tf.random.uniform(), trainable=False)

    # 2nd call student ------------------------------
    logits['s_on_l_new'] = student(l_images, training=True)
    cross_entroy['s_on_l_new'] = tf.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM
    )