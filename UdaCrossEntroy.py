import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

import config
from Model import Wrn28k


def UdaCrossEntroy(all_logits, l_labels, global_step):
    batch_size = config.BATCH_SIZE
    uda_data = config.UDA_DATA
    logits = {}
    labels = {}
    cross_entroy = {}
    masks = {}
    # 将网络的输出结果区分成 label ori aug 三个部分
    logits['l'], logits['ori'], logits['aug'] = tf.split(
        all_logits,
        [batch_size, batch_size * uda_data, batch_size * uda_data],
        axis=0,
    )
    # 对标签进行处理
    labels['l'] = l_labels

    # ------------loss的计算---------
    # part1：有监督部分
    cross_entroy['l'] = keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=config.LABEL_SMOOTHING,
        reduction=keras.losses.Reduction.NONE,
    )(labels['l'], logits['l'])
    '''
    probs = tf.nn.softmax(logits['l'], axis=-1)  # 将每张图片对应10个类别的输出转化为概率的形式
    correct_probs = tf.reduce_sum(labels['l'] * probs, axis=-1)  # 根据图片对应的label和概率计算出 预测正确类别的概率

    # 计算一个阈值l_threshold
    r = tf.cast(global_step, tf.float32) / tf.convert_to_tensor(config.MAX_STEPS, dtype=tf.float32)
    num_classes = tf.convert_to_tensor(config.NUM_CLASSES, tf.float32)
    l_threshold = r * (1. - 1. / num_classes) + 1. / num_classes

    masks['l'] = tf.math.less_equal(correct_probs, l_threshold)
    masks['l'] = tf.cast(masks['l'], tf.float32)
    masks['l'] = tf.stop_gradient(masks['l'])  # 如果对某图片预测的概率小于l_threahold,输出1，否则是0
    '''
    cross_entroy['l'] = tf.reduce_sum(cross_entroy['l']) / float(batch_size)

    # part2: 无监督部分
    labels['ori'] = tf.nn.softmax(logits['ori'] / tf.convert_to_tensor(config.UDA_TEMP), axis=-1)
    labels['ori'] = tf.stop_gradient(labels['ori'])
    # tf.nn.log_softmax: 设一张图片对应3个类别的输出为o1，o2，o3 ==>
    # b = log(sum(exp(o1) + exp(o2) + exp(o3)))  new_o1=o1-b, new_o2=o2-b ... 恒负，大小关系不变
    cross_entroy['u'] = (
            labels['ori'] * tf.nn.log_softmax(logits['aug'], axis=-1)
    )

    largest_probs = tf.reduce_max(labels['ori'], axis=-1, keepdims=True)

    masks['u'] = tf.math.greater_equal(largest_probs, tf.constant(config.UDA_THRESHOLD))  # 判断最大概率是否大于阈值
    masks['u'] = tf.cast(masks['u'], config.DTYPE)
    masks['u'] = tf.stop_gradient(masks['u'])
    # 极端情况，当ori的预测完全准确，即class i = 1, 其他类别为0时，
    # aug的class i最大，即最大的负数，两者相乘再取负，就是一个非常接近于0的数字
    cross_entroy['u'] = tf.reduce_sum(-cross_entroy['u'] * masks['u']) / \
                        tf.convert_to_tensor((batch_size * uda_data), dtype=config.DTYPE)

    return logits, labels, masks, cross_entroy


if __name__ == '__main__':
    # 制作数据
    l_images = np.random.random((1, 32, 32, 3))
    l_images = tf.convert_to_tensor(l_images, dtype=config.DTYPE)
    ori_images = np.random.random((1 * config.UDA_DATA, 32, 32, 3))
    ori_images = tf.convert_to_tensor(ori_images, dtype=config.DTYPE)
    aug_images = np.random.random((1 * config.UDA_DATA, 32, 32, 3))
    aug_images = tf.convert_to_tensor(aug_images, dtype=config.DTYPE)
    all_images = tf.concat([l_images, ori_images, aug_images], axis=0)  # shape [3, 32, 32, 3]

    l_labels = np.array([2])
    l_labels = tf.convert_to_tensor(l_labels, dtype=tf.int32)
    l_labels = tf.raw_ops.OneHot(indices=l_labels, depth=config.NUM_CLASSES, on_value=1.0, off_value=0)
    l_labels = tf.cast(l_labels, config.DTYPE)

    # 构建teacher模型，产生输出
    teacher = Wrn28k(num_inp_filters=3, k=2)
    output = teacher(x=all_images)  # shape=[15, 10]

    logits, labels, masks, cross_entroy = UdaCrossEntroy(output, l_labels, 1)
    print('logits: ', logits.keys())
    print('labels: ', labels.keys())
    print('masks: ', masks.keys())
    # print('cross entroy: ', cross_entroy)
