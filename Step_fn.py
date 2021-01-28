import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from UdaCrossEntroy import UdaCrossEntroy


import config
from Model import Wrn28k


if __name__ == '__main__':
    uda_data = int(config.UDA_DATA)
    batch_size = 1
    # 制作数据集
    l_images = np.random.random((1, 32, 32, 3))
    l_images = tf.convert_to_tensor(l_images, dtype=tf.float32)
    ori_images = np.random.random((1 * config.UDA_DATA, 32, 32, 3))
    ori_images = tf.convert_to_tensor(ori_images, dtype=tf.float32)
    aug_images = np.random.random((1 * config.UDA_DATA, 32, 32, 3))
    aug_images = tf.convert_to_tensor(aug_images, dtype=tf.float32)

    all_images = tf.concat([l_images, ori_images, aug_images], axis=0)  # shape [15, 32, 32, 3]
    u_aug_and_l_images = tf.concat([aug_images, l_images], axis=0)
    # 标签
    l_labels = np.array([2])
    l_labels = tf.convert_to_tensor(l_labels, dtype=tf.int32)
    l_labels = tf.raw_ops.OneHot(indices=l_labels, depth=config.NUM_CLASSES, on_value=1.0, off_value=0)
    # print(l_labels, l_labels.shape)

    # 构建teacher模型
    teacher = Wrn28k(num_inp_filters=3, k=2)
    # 构建student模型
    student = Wrn28k(num_inp_filters=3, k=2)

    # 定义student的损失函数， PS：teacher的损失函数为UdaCrossEntroy
    s_unlabel_loss = tf.losses.CategoricalCrossentropy(
        label_smoothing=config.LABEL_SMOOTHING,
        from_logits=False,
        reduction=tf.keras.losses.Reduction.NONE,
    )

    s_label_loss = tf.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM,
        from_logits=False,
    )

    # 整个流程----------------------
    # step1：经过teacher，得到输出
    output = teacher(x=all_images)  # shape=[15, 10]
    logits, labels, masks, cross_entroy = UdaCrossEntroy(output, l_labels, 10)
    # ------打印teacher经过loss的输出------
    # print('logits: ', logits.keys(), type(logits))
    # print('labels: ', labels.keys(), type(labels))
    # print('masks: ', masks.keys(), type(masks))
    # print('cross entroy: ', cross_entroy.keys(), type(cross_entroy))

    # step2：第一次call student -----------------------------
    logits['s_on_aug_and_l'] = student(x=u_aug_and_l_images)  # shape=[8, 10]
    logits['s_on_u'], logits['s_on_l_old'] = tf.split(
        logits['s_on_aug_and_l'],
        [aug_images.shape[0], l_images.shape[0]],
        axis=0
    )
    # print(logits['s_on_u'].shape, logits['s_on_l_old'].shape)
    cross_entroy['s_on_u'] = s_unlabel_loss(
        y_true=tf.stop_gradient(tf.nn.softmax(logits['aug'], -1)),
        y_pred=logits['s_on_u']
    )
    cross_entroy['s_on_u'] = tf.reduce_sum(cross_entroy['s_on_u']) / \
                             tf.convert_to_tensor(batch_size*uda_data, dtype=tf.float32)

    # for taylor
    cross_entroy['s_on_l_old'] = s_label_loss(
        y_true=labels['l'],
        y_pred=logits['s_on_l_old']
    )
    cross_entroy['s_on_l_old'] = tf.reduce_sum(cross_entroy['s_on_l_old']) / \
                                 tf.convert_to_tensor(batch_size, dtype=tf.float32)

    shadow = tf.Variable(
        initial_value=cross_entroy['s_on_l_old'],
        trainable=False,
        name='cross_entroy_old'
    )
    shadow_update = tf.Variable(
        initial_value=cross_entroy['s_on_l_old'],
        trainable=False,
        name='shadow_update'
    )
    # shadow_update = shadow_update.assign(shadow)
    print(shadow)
    print(shadow_update)

    # 2nd call student ------------------------------
    logits['s_on_l_new'] = student(l_images, training=True)
    cross_entroy['s_on_l_new'] = s_label_loss(
        y_true=labels['l'],
        y_pred=logits['s_on_l_new']
    )
    cross_entroy['s_on_l_new'] = tf.reduce_sum(cross_entroy['s_on_l_new']) / \
                                 tf.convert_to_tensor(batch_size, dtype=tf.float32)