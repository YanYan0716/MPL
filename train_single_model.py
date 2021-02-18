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

    # 构建模型
    teacher = Wrn28k(num_inp_filters=3, k=2)

    # 定义损失函数，
    t_label_loss = tf.losses.CategoricalCrossentropy(
        # reduction=tf.keras.losses.Reduction.NONE,
        from_logits=True,
    )

    # 定义学习率
    Tea_lr_fun = LearningRate(
        config.TEACHER_LR,
        config.TEACHER_LR_WARMUP_STEPS,
        config.TEACHER_NUM_WAIT_STEPS
    )

    global_step = 0

    for epoch in range(config.MAX_EPOCHS):
        SLOSS = 0
        for batch_idx, (data) in enumerate(ds_label_train):
            teacher.training = True
            global_step += 1
            l_images = data['images']
            l_labels = data['labels']
            with tf.GradientTape() as s_tape:
                logits = teacher(x=l_images)  # shape=[8, 10]
                cross_entroy = t_label_loss(
                    y_true=l_labels,
                    y_pred=logits,
                )
                # 计算损失函数
                # cross_entroy = tf.reduce_sum(cross_entroy) / \
                #                          tf.convert_to_tensor(config.BATCH_SIZE, dtype=tf.float32)
                SLOSS += cross_entroy
            # 反向传播，更新参数-------
            TeacherLR = Tea_lr_fun.__call__(global_step=global_step)
            # TeaOptim = keras.optimizers.SGD(
            #     learning_rate=TeacherLR,
            #     momentum=0.9,
            #     # nesterov=True,
            # )
            TeaOptim = keras.optimizers.Adam(lr=3e-4)
            GStud_unlabel = s_tape.gradient(cross_entroy, teacher.trainable_variables)
            # GStud_unlabel, _ = tf.clip_by_global_norm(GStud_unlabel, config.GRAD_BOUND)
            TeaOptim.apply_gradients(zip(GStud_unlabel, teacher.trainable_variables))

            if (batch_idx + 1) % config.LOG_EVERY == 0:
                SLOSS = SLOSS / config.LOG_EVERY
                print(f'global: %4d' % global_step + ',[epoch:%4d/' % epoch + 'EPOCH: %4d] \t' % config.MAX_EPOCHS
                      + '/[Loss: %.4f]' % SLOSS + ' [LR: %.6f]' % TeacherLR + '   %2d' % len(
                    teacher.trainable_variables))
                SLOSS = 0
        # 测试test上的acc
        if (TeacherLR > 0) and (epoch % 5 == 0):
            acc = test(teacher)
            print(f'testing ... acc: {acc}')
        # 保存weights
        # if ((epoch + 1) % config.SAVE_EVERY == 0) and (TeacherLR > 0):
        #     Ssave_path = config.STD_SAVE_PATH + str(epoch + 1) + '_' + str(batch_idx + 1)
        #     tf.saved_model.save(teacher, Ssave_path)
        #     print(f'saving for epoch {epoch}, Spath:{Ssave_path}')
