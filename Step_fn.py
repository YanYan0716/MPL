import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from UdaCrossEntroy import UdaCrossEntroy


import config
from Model import Wrn28k


if __name__ == '__main__':
    uda_data = int(config.UDA_DATA)
    batch_size = 2
    # np.random.seed(1)
    # 制作数据集
    l_images = np.random.random((batch_size, 32, 32, 3))
    l_images = tf.convert_to_tensor(l_images, dtype=tf.float32)
    ori_images = np.random.random((batch_size * config.UDA_DATA, 32, 32, 3))
    ori_images = tf.convert_to_tensor(ori_images, dtype=tf.float32)
    aug_images = np.random.random((batch_size * config.UDA_DATA, 32, 32, 3))
    aug_images = tf.convert_to_tensor(aug_images, dtype=tf.float32)

    all_images = tf.concat([l_images, ori_images, aug_images], axis=0)  # shape [15, 32, 32, 3]
    u_aug_and_l_images = tf.concat([aug_images, l_images], axis=0)
    # 标签
    l_labels = np.array([2, 3])
    l_labels = tf.convert_to_tensor(l_labels, dtype=tf.int32)
    l_labels = tf.raw_ops.OneHot(indices=l_labels, depth=config.NUM_CLASSES, on_value=1.0, off_value=0)
    # print(l_labels, l_labels.shape)

    # 构建teacher模型
    teacher = Wrn28k(num_inp_filters=3, k=2)
    # 构建student模型
    student = Wrn28k(num_inp_filters=3, k=2)

    # 定义teacher的损失函数，损失函数之一为UdaCrossEntroy
    mpl_loss = tf.losses.CategoricalCrossentropy(
        reduction=tf.losses.Reduction.NONE
    )
    # 定义student的损失函数， PS：teacher的损失函数为UdaCrossEntroy
    s_unlabel_loss = tf.losses.CategoricalCrossentropy(
        label_smoothing=config.LABEL_SMOOTHING,
        from_logits=False,
        reduction=tf.keras.losses.Reduction.NONE,
    )

    s_label_loss = tf.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE,
        from_logits=False,
    )

    # 定义teacher的优化函数
    TeaOptim = keras.optimizers.SGD(learning_rate=0.0)
    # 定义student的优化函数
    StdOptim = keras.optimizers.SGD(learning_rate=0.0)

    # 整个流程----------------------
    # step1：经过teacher，得到输出
    with tf.GradientTape() as t_tape:
        output = teacher(x=all_images)  # shape=[15, 10]
        logits, labels, masks, cross_entroy = UdaCrossEntroy(output, l_labels, 10)
    # ------打印teacher经过loss的输出------
    # print('logits: ', logits.keys(), type(logits))
    # print('labels: ', labels.keys(), type(labels))
    # print('masks: ', masks.keys(), type(masks))
    # print('cross entroy: ', cross_entroy.keys(), type(cross_entroy))

    # step2：第一次call student -----------------------------
    with tf.GradientTape() as s_tape:
        logits['s_on_aug_and_l'] = student(x=u_aug_and_l_images)  # shape=[8, 10]
        logits['s_on_u'], logits['s_on_l_old'] = tf.split(
            logits['s_on_aug_and_l'],
            [aug_images.shape[0], l_images.shape[0]],
            axis=0
        )

        cross_entroy['s_on_u'] = s_unlabel_loss(
            y_true=tf.stop_gradient(tf.nn.softmax(logits['aug'], -1)),
            y_pred=logits['s_on_u']
        )
        # 计算损失函数
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
        logits['test'] = student(l_images)
        cross_entroy['test'] = s_label_loss(
            y_true=labels['l'],
            y_pred=logits['test']
        )
        cross_entroy['test'] = tf.reduce_sum(cross_entroy['test']) / \
                                     tf.convert_to_tensor(batch_size, dtype=tf.float32)

    # 反向传播，更新student的参数-------
    GStud_unlabel = s_tape.gradient(cross_entroy['s_on_u'], student.trainable_variables)
    StdOptim.apply_gradients(zip(GStud_unlabel, student.trainable_variables))

    # step3: 2nd call student ------------------------------
    logits['s_on_l_new'] = student(l_images)
    cross_entroy['s_on_l_new'] = s_label_loss(
        y_true=labels['l'],
        y_pred=logits['s_on_l_new']
    )
    cross_entroy['s_on_l_new'] = tf.reduce_sum(cross_entroy['s_on_l_new']) / \
                                 tf.convert_to_tensor(batch_size, dtype=tf.float32)

    dot_product = cross_entroy['s_on_l_new'] - shadow
    moving_dot_product = keras.initializers.GlorotNormal()(shape=dot_product.shape)
    moving_dot_product = tf.Variable(initial_value=moving_dot_product, trainable=False, dtype=tf.float32)
    moving_dot_product_update = moving_dot_product.assign_sub(0.01*(moving_dot_product-dot_product))
    dot_product = dot_product - moving_dot_product
    dot_product = tf.stop_gradient(dot_product)
    print('ok')
    # print(cross_entroy['s_on_l_old'])
    print(cross_entroy['s_on_l_new'])
    print(cross_entroy['test'])
    # step4: 求teacher的损失函数
    # cross_entroy['mpl'] = mpl_loss(
    #     y_true=tf.stop_gradient(tf.nn.softmax(logits['aug'], axis=-1)),
    #     y_pred=logits['aug']
    # )
    # cross_entroy['mpl'] = tf.reduce_sum(cross_entroy['mpl'])/\
    #                       tf.convert_to_tensor(float(batch_size*uda_data), dtype=tf.float32)
    # uda_weight = config.UDA_WEIGHT * tf.math.minimum(
    #     1., tf.cast(config.GLOBAL_STEP, tf.float32)/float(config.UDA_STEPS)
    # )
    # teacher_loss = cross_entroy['u']*config.UDA_WEIGHT + \
    #                cross_entroy['l'] + \
    #                cross_entroy['mpl']*dot_product
    # # 反向传播，更新teacher的参数-------
    # GTea = s_tape.gradient(teacher_loss, teacher.trainable_variables)
    # TeaOptim.apply_gradients(zip(GTea, teacher.trainable_variables))