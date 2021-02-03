import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import pandas as pd

import config
from Model import Wrn28k
from UdaCrossEntroy import UdaCrossEntroy
from learningRate import LearningRate
from Dataset import label_image
from Dataset import unlabel_image
from Dataset import merge_dataset

if __name__ == '__main__':
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # 有标签的数据集 batch_size=config.BATCH_SIZE
    df_label = pd.read_csv(config.LABEL_FILE_PATH)
    file_paths = df_label['file_name'].values
    labels = df_label['label'].values
    ds_label_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_label_train = ds_label_train \
        .map(label_image, num_parallel_calls=AUTOTUNE) \
        .batch(config.BATCH_SIZE)

    # 无标签的数据集 batch_size=config.BATCH_SIZE*config.UDA_DATA
    df_unlabel = pd.read_csv(config.UNLABEL_FILE_PATH)
    file_paths = df_unlabel['file_name'].values
    labels = df_unlabel['label'].values
    ds_unlabel_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_unlabel_train = ds_unlabel_train \
        .map(unlabel_image, num_parallel_calls=AUTOTUNE) \
        .batch(config.BATCH_SIZE * config.UDA_DATA)

    # 将有标签数据和无标签数据整合成最终的数据形式
    ds_train = tf.data.Dataset.zip((ds_label_train, ds_unlabel_train))
    ds_train = ds_train.map(merge_dataset)

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

    # 定义teacher的学习率
    Tea_lr_fun = LearningRate(
        config.TEACHER_LR,
        config.TEACHER_LR_WARMUP_STEPS,
        config.TEACHER_NUM_WAIT_STEPS
    )
    # 定义student的学习率
    Std_lr_fun = LearningRate(
        config.STUDENT_LR,
        config.STUDENT_LR_WARMUP_STEPS,
        config.STUDENT_LR_WAIT_STEPS
    )

    global_step = 0

    for epoch in range(config.MAX_EPOCHS):
        for batch_idx, (l_images, l_labels, ori_images, aug_images) in enumerate(ds_train):
            global_step += 1
            all_images = tf.concat([l_images, ori_images, aug_images], axis=0)  # shape [15, 32, 32, 3]
            u_aug_and_l_images = tf.concat([aug_images, l_images], axis=0)
            # step1：经过teacher，得到输出
            with tf.GradientTape() as t_tape:
                output = teacher(x=all_images)  # shape=[15, 10]
                logits, labels, masks, cross_entroy = UdaCrossEntroy(output, l_labels, global_step)
            # step2：1st call student -----------------------------
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
                                         tf.convert_to_tensor(config.BATCH_SIZE * config.UDA_DATA, dtype=tf.float32)
                # for taylor
                cross_entroy['s_on_l_old'] = s_label_loss(
                    y_true=labels['l'],
                    y_pred=logits['s_on_l_old']
                )

                cross_entroy['s_on_l_old'] = tf.reduce_sum(cross_entroy['s_on_l_old']) / \
                                             tf.convert_to_tensor(config.BATCH_SIZE, dtype=tf.float32)
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
            # 反向传播，更新student的参数-------
            StudentLR = Std_lr_fun.__call__(global_step=global_step)
            StdOptim = keras.optimizers.SGD(learning_rate=StudentLR)
            GStud_unlabel = s_tape.gradient(cross_entroy['s_on_u'], student.trainable_variables)
            StdOptim.apply_gradients(zip(GStud_unlabel, student.trainable_variables))
            # step3: 2nd call student ------------------------------
            logits['s_on_l_new'] = student(l_images)
            cross_entroy['s_on_l_new'] = s_label_loss(
                y_true=labels['l'],
                y_pred=logits['s_on_l_new']
            )
            cross_entroy['s_on_l_new'] = tf.reduce_sum(cross_entroy['s_on_l_new']) / \
                                         tf.convert_to_tensor(config.BATCH_SIZE, dtype=tf.float32)
            dot_product = cross_entroy['s_on_l_new'] - shadow
            moving_dot_product = keras.initializers.GlorotNormal()(shape=dot_product.shape)
            moving_dot_product = tf.Variable(initial_value=moving_dot_product, trainable=False, dtype=tf.float32)
            moving_dot_product_update = moving_dot_product.assign_sub(0.01 * (moving_dot_product - dot_product))
            dot_product = dot_product - moving_dot_product
            dot_product = tf.stop_gradient(dot_product)
            # step4: 求teacher的损失函数
            with t_tape:
                cross_entroy['mpl'] = mpl_loss(
                    y_true=tf.stop_gradient(tf.nn.softmax(logits['aug'], axis=-1)),
                    y_pred=logits['aug']
                )
                cross_entroy['mpl'] = tf.reduce_sum(cross_entroy['mpl']) / \
                                      tf.convert_to_tensor(config.BATCH_SIZE * config.UDA_DATA, dtype=tf.float32)
                uda_weight = config.UDA_WEIGHT * tf.math.minimum(
                    1., tf.cast(global_step, tf.float32) / float(config.UDA_STEPS)
                )
                teacher_loss = cross_entroy['u'] * uda_weight + \
                               cross_entroy['l'] + \
                               cross_entroy['mpl'] * dot_product
            # 反向传播，更新teacher的参数-------
            TeacherLR = Tea_lr_fun.__call__(global_step=global_step)
            TeaOptim = keras.optimizers.SGD(learning_rate=TeacherLR)
            GTea = t_tape.gradient(teacher_loss, teacher.trainable_variables)
            TeaOptim.apply_gradients(zip(GTea, teacher.trainable_variables))

            if batch_idx % config.LOG_EVERY == 0:
                print(f'batch: %3d' % batch_idx + ',[epoch:%4d/' % epoch + 'EPOCH: %4d] \t' % config.MAX_EPOCHS
                      + '[Teacher Loss: %.4f]' % teacher_loss + '/[Student Loss: %.4f]' % cross_entroy['s_on_u']
                      + '\t[Teacher LR: %.6f' % TeacherLR + ']/[Student LR: %.6f]' % StudentLR)
            # if batch_idx % config.SAVE_EVERY == 0:
            #     Tcheckpoint_prefix = config.TEA_SAVE_PATH + '/ckpt'
            #     Scheckpoint_prefix = config.STD_SAVE_PATH + '/ckpt'
            #
            #     Tcheckpoint = tf.train.Checkpoint(model=teacher, optimizer=TeaOptim)
            #     Scheckpoint = tf.train.Checkpoint(model=student, optimizer=StdOptim)
            #
            #     Tstatus = Tcheckpoint.restore(tf.train.latest_checkpoint(config.TEA_SAVE_PATH))
            #     Sstatus = Tcheckpoint.restore(tf.train.latest_checkpoint(config.STD_SAVE_PATH))
            #
            #     Tstatus.assert_consumed()
            #     Tcheckpoint.save(Tcheckpoint_prefix)
            #
            #     Sstatus.assert_consumed()
            #     Scheckpoint.save(Scheckpoint_prefix)
            #     print('saving checkpoint ...')
