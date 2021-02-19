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
from test import test

if __name__ == '__main__':
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # 有标签的数据集 batch_size=config.BATCH_SIZE
    df_label = pd.read_csv(config.LABEL_FILE_PATH)
    file_paths = df_label['file_name'].values
    labels = df_label['label'].values
    ds_label_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_label_train = ds_label_train \
        .map(label_image, num_parallel_calls=AUTOTUNE) \
        .batch(config.BATCH_SIZE, drop_remainder=True)\
        .shuffle(buffer_size=config.SHUFFLE_SIZE)

    # 无标签的数据集 batch_size=config.BATCH_SIZE*config.UDA_DATA
    df_unlabel = pd.read_csv(config.UNLABEL_FILE_PATH)
    file_paths = df_unlabel['file_name'].values
    labels = df_unlabel['label'].values
    ds_unlabel_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_unlabel_train = ds_unlabel_train \
        .map(unlabel_image, num_parallel_calls=AUTOTUNE) \
        .batch(config.BATCH_SIZE * config.UDA_DATA, drop_remainder=True)\
        .shuffle(buffer_size=config.SHUFFLE_SIZE*config.UDA_DATA)

    # 将有标签数据和无标签数据整合成最终的数据形式
    ds_train = tf.data.Dataset.zip((ds_label_train, ds_unlabel_train))
    ds_train = ds_train.map(merge_dataset)

    # 构建teacher模型
    teacher = Wrn28k(num_inp_filters=3, k=2)

    # 定义teacher的损失函数，损失函数之一为UdaCrossEntroy
    mpl_loss = tf.losses.CategoricalCrossentropy(
        reduction=tf.losses.Reduction.NONE
    )

    # 定义teacher的学习率
    Tea_lr_fun = LearningRate(
        config.TEACHER_LR,
        config.TEACHER_LR_WARMUP_STEPS,
        config.TEACHER_NUM_WAIT_STEPS
    )
    global_step = 0

    for epoch in range(config.MAX_EPOCHS):
        TLOSS = 0
        TLOSS_1 = 0
        TLOSS_2 = 0
        TLOSS_3 = 0
        for batch_idx, (l_images, l_labels, ori_images, aug_images) in enumerate(ds_train):
            global_step += 1
            all_images = tf.concat([l_images, ori_images, aug_images], axis=0)  # shape [15, 32, 32, 3]
            u_aug_and_l_images = tf.concat([aug_images, l_images], axis=0)
            # step1：经过teacher，得到输出
            with tf.GradientTape() as t_tape:
                output = teacher(x=all_images)  # shape=[15, 10]
                logits, labels, masks, cross_entroy = UdaCrossEntroy(output, l_labels, global_step)

            # step4: 求teacher的损失函数
            with t_tape:
                uda_weight = config.UDA_WEIGHT * tf.math.minimum(
                    1., tf.cast(global_step, tf.float32) / float(config.UDA_STEPS)
                )
                teacher_loss = cross_entroy['u'] * uda_weight + cross_entroy['l']
                TLOSS += teacher_loss
                TLOSS_1 += (cross_entroy['u'] * uda_weight)
                TLOSS_2 += cross_entroy['l']
            # 反向传播，更新teacher的参数-------
            TeacherLR = Tea_lr_fun.__call__(global_step=global_step)
            TeaOptim = keras.optimizers.SGD(
                learning_rate=TeacherLR,
                momentum=0.9,
                nesterov=True,
            )
            GTea = t_tape.gradient(teacher_loss, teacher.trainable_variables)
            GTea, _ =  tf.clip_by_global_norm(GTea, config.GRAD_BOUND)
            TeaOptim.apply_gradients(zip(GTea, teacher.trainable_variables))

            if (batch_idx + 1) % config.LOG_EVERY == 0:
                TLOSS = TLOSS / config.LOG_EVERY
                TLOSS_1 = TLOSS_1 / config.LOG_EVERY
                TLOSS_2 = TLOSS_2 / config.LOG_EVERY
                TLOSS_3 = TLOSS_3 / config.LOG_EVERY
                SLOSS = SLOSS / config.LOG_EVERY
                print(f'global: %4d' % global_step + ',[epoch:%4d/' % epoch + 'EPOCH: %4d] \t' % config.MAX_EPOCHS
                      + '[U:%.4f' % (TLOSS_1) + ', L:%.4f' % (TLOSS_2) + ', M:%.4f' % (
                          TLOSS_3) + ']' + '[TLoss: %.4f]' % TLOSS
                      + '\t[TLR: %.6f' % TeacherLR + ']')
                TLOSS = 0
                TLOSS_1 = 0
                TLOSS_2 = 0
                TLOSS_3 = 0
        # 测试teacher在test上的acc
        if epoch % 5 == 0:
            Tacc = test(teacher)
            print(f'testing teacher model ... acc: {Tacc}')