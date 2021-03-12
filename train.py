import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import tensorflow_addons as tfa
from WideResnet import WideResnet
from copy import deepcopy


import config
from Model import Wrn28k
from UdaCrossEntroy import UdaCrossEntroy
from learningRate import LearningRate
from Dataset import label_image
from Dataset import unlabel_image
from Dataset import merge_dataset
from test import test


def my_update(model, model_):
    for i in range(len(model_)):
        model.weights[i] = model.weights[i].assign(
            model.weights[i]*(1-config.EMA)+model_[i]*config.EMA)
    model_ = deepcopy(model.weights)
    return model, model_


if __name__ == '__main__':
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # 有标签的数据集 batch_size=config.BATCH_SIZE
    df_label = pd.read_csv(config.LABEL_FILE_PATH)
    file_paths = df_label['file_name'].values
    labels = df_label['label'].values
    ds_label_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_label_train = ds_label_train \
        .map(label_image, num_parallel_calls=AUTOTUNE) \
        .batch(config.BATCH_SIZE, drop_remainder=True) \
        .shuffle(buffer_size=4000)\
        .prefetch(AUTOTUNE)

    # 无标签的数据集 batch_size=config.BATCH_SIZE*config.UDA_DATA
    df_unlabel = pd.read_csv(config.UNLABEL_FILE_PATH)
    file_paths = df_unlabel['name'].values
    labels = df_unlabel['label'].values
    ds_unlabel_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_unlabel_train = ds_unlabel_train \
        .map(unlabel_image, num_parallel_calls=AUTOTUNE) \
        .batch(config.BATCH_SIZE * config.UDA_DATA, drop_remainder=True) \
        .shuffle(buffer_size=50000)\
        .prefetch(AUTOTUNE)

    # 将有标签数据和无标签数据整合成最终的数据形式
    ds_train = tf.data.Dataset.zip((ds_label_train, ds_unlabel_train))
    ds_train = ds_train.map(merge_dataset)

    # 构建teacher模型
    if config.TEA_CONTINUE:
        print('continue teacher training')
        teacher = WideResnet().model()
        teacher.load_weights(config.TEA_LOAD_PATH)
        teacher.training = True
    else:
        # teacher = Wrn28k(num_inp_filters=3, k=2)
        teacher = WideResnet().model()

    # 构建student模型
    if config.STD_CONTINUE:
        print('continue student training')
        student = WideResnet().model()
        student.load_weights(config.STD_LOAD_PATH)
        student.training = True
        # student = tf.saved_model.load(config.STD_LOAD_PATH)
    else:
        # student = Wrn28k(num_inp_filters=3, k=2)
        student = WideResnet().model()
    student_ = student.weights

    # 定义teacher的损失函数，损失函数之一为UdaCrossEntroy
    mpl_loss = tf.losses.CategoricalCrossentropy(
        reduction=tf.losses.Reduction.NONE,
        from_logits=True,
    )
    # 定义student的损失函数， PS：teacher的损失函数为UdaCrossEntroy
    s_unlabel_loss = tf.losses.CategoricalCrossentropy(
        label_smoothing=config.LABEL_SMOOTHING,
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE,
    )

    s_label_loss = tf.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE,
        from_logits=True,
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

    global_step = 62*config.CONTINUE_EPOCH
    print(f'start training from global step {global_step}......')
    TBacc = 0
    Tacc = 0
    SBacc = 0
    Sacc = 0
    epochs = config.MAX_EPOCHS - config.CONTINUE_EPOCH
    for epoch in range(epochs):
        TLOSS = 0
        TLOSS_1 = 0
        TLOSS_2 = 0
        TLOSS_3 = 0
        SLOSS = 0
        for batch_idx, (l_images, l_labels, ori_images, aug_images) in enumerate(ds_train):
            global_step += 1
            all_images = tf.concat([l_images, ori_images, aug_images], axis=0)  # shape [15, 32, 32, 3]
            u_aug_and_l_images = tf.concat([aug_images, l_images], axis=0)
            # step1：经过teacher，得到输出
            with tf.GradientTape() as t_tape:
                output = teacher(all_images)  # shape=[15, 10]
                logits, labels, masks, cross_entroy = UdaCrossEntroy(output, l_labels, global_step)
            # step2：1st call student -----------------------------
            with tf.GradientTape() as s_tape:
                logits['s_on_aug_and_l'] = student(u_aug_and_l_images)  # shape=[8, 10]
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
                SLOSS += cross_entroy['s_on_u']
                # for taylor
                cross_entroy['s_on_l_old'] = s_label_loss(
                    y_true=labels['l'],
                    y_pred=logits['s_on_l_old']
                )

                cross_entroy['s_on_l_old'] = tf.reduce_sum(cross_entroy['s_on_l_old']) / \
                                             tf.convert_to_tensor(config.BATCH_SIZE, dtype=tf.float32)
            # 反向传播，更新student的参数-------
            StudentLR = Std_lr_fun.__call__(global_step=global_step)
            StdOptim = keras.optimizers.SGD(
                learning_rate=StudentLR,
                momentum=0.9,
                nesterov=True,
                # weight_decay=5e-4,
            )
            # StdOptim = keras.optimizers.Adam(learning_rate=StudentLR)
            GStud_unlabel = s_tape.gradient(cross_entroy['s_on_u'], student.trainable_variables)
            GStud_unlabel, _ = tf.clip_by_global_norm(GStud_unlabel, config.GRAD_BOUND)
            StdOptim.apply_gradients(zip(GStud_unlabel, student.trainable_variables))
            # 如何更新参数
            student, student_ = my_update(student, student_)

            # step3: 2nd call student ------------------------------
            logits['s_on_l_new'] = student(l_images)
            cross_entroy['s_on_l_new'] = s_label_loss(
                y_true=labels['l'],
                y_pred=logits['s_on_l_new']
            )
            cross_entroy['s_on_l_new'] = tf.reduce_sum(cross_entroy['s_on_l_new']) / \
                                         tf.convert_to_tensor(config.BATCH_SIZE, dtype=config.DTYPE)
            dot_product = cross_entroy['s_on_l_new'] - cross_entroy['s_on_l_old']
            limit = 3.0**(0.5)
            moving_dot_product = tf.random_uniform_initializer(minval=-limit, maxval=limit)(shape=dot_product.shape)
            moving_dot_product = tf.Variable(initial_value=moving_dot_product, trainable=False, dtype=config.DTYPE)
            moving_dot_product_update = moving_dot_product.assign_sub(0.01 * (moving_dot_product - dot_product))
            dot_product = dot_product - moving_dot_product
            dot_product = tf.stop_gradient(dot_product)
            # step4: 求teacher的损失函数
            with t_tape:
                # label = tf.math.argmax(tf.nn.softmax(logits['aug'], axis=-1), axis=-1)
                # label = tf.raw_ops.OneHot(indices=label, depth=config.NUM_CLASSES, on_value=1.0, off_value=0)
                cross_entroy['mpl'] = mpl_loss(
                    # y_true=tf.stop_gradient(label),
                    y_true=tf.stop_gradient(tf.nn.softmax(logits['aug'], axis=-1)),
                    y_pred=logits['aug']
                )  # 恒正
                cross_entroy['mpl'] = tf.reduce_sum(cross_entroy['mpl']) / \
                                      tf.convert_to_tensor(config.BATCH_SIZE * config.UDA_DATA, dtype=config.DTYPE)
                uda_weight = config.UDA_WEIGHT * tf.math.minimum(
                    1., tf.cast(global_step, config.DTYPE) / float(config.UDA_STEPS)
                )
                # if StudentLR == 0:
                #     dot_product = 0
                teacher_loss = cross_entroy['u'] * uda_weight + \
                               cross_entroy['l'] + \
                               cross_entroy['mpl'] * dot_product

                TLOSS += teacher_loss
                TLOSS_1 += (cross_entroy['u'] * uda_weight)
                TLOSS_2 += cross_entroy['l']
                TLOSS_3 += cross_entroy['mpl'] * dot_product
            # 反向传播，更新teacher的参数-------
            TeacherLR = Tea_lr_fun.__call__(global_step=global_step)
            TeaOptim = keras.optimizers.SGD(
                learning_rate=TeacherLR,
                momentum=0.9,
                nesterov=True,
                # weight_decay=5e-4,
            )
            # TeaOptim = keras.optimizers.Adam(learning_rate=TeacherLR)
            GTea = t_tape.gradient(teacher_loss, teacher.trainable_variables)
            GTea, _ = tf.clip_by_global_norm(GTea, config.GRAD_BOUND)
            TeaOptim.apply_gradients(zip(GTea, teacher.trainable_variables))

            if (batch_idx + 1) % config.LOG_EVERY == 0:
                TLOSS = TLOSS / config.LOG_EVERY
                TLOSS_1 = TLOSS_1 / config.LOG_EVERY
                TLOSS_2 = TLOSS_2 / config.LOG_EVERY
                TLOSS_3 = TLOSS_3 / config.LOG_EVERY
                SLOSS = SLOSS / config.LOG_EVERY
                print(f'global: %4d' % global_step + ',[epoch:%4d/' % (epoch+config.CONTINUE_EPOCH) + 'EPOCH: %4d] \t' % epochs
                      + '[U:%.4f' % (TLOSS_1) + ', L:%.4f' % (TLOSS_2) + ', M:%.4f' % (
                          TLOSS_3) + ']' + '[TLoss: %.4f]' % TLOSS + '/[SLoss: %.4f]' % SLOSS
                      + '\t[TLR: %.6f' % TeacherLR + ']/[SLR: %.6f]' % StudentLR)
                TLOSS = 0
                TLOSS_1 = 0
                TLOSS_2 = 0
                TLOSS_3 = 0
                SLOSS = 0
        # 测试teacher在test上的acc
        if epoch % 5 == 0:
            Tacc = test(teacher)
            print(f'testing teacher model ... acc: {Tacc}')
        # 测试student在test上的acc，当student开始训练的时候
        if (StudentLR > 0) and (epoch % 5 == 0):
            Sacc = test(student)
            print(f'testing ... acc: {Sacc}')
        # 保存weights
        if Tacc > TBacc:
            Tsave_path = config.TEA_SAVE_PATH  # + str(epoch + 1) + '_' + str(batch_idx + 1)
            teacher.save_weights(Tsave_path)
            # tf.saved_model.save(teacher, Tsave_path)
            TBacc = Tacc
            print(f'saving for TBacc {TBacc}, Tpath:{Tsave_path}')
        if Sacc > SBacc:
            Ssave_path = config.STD_SAVE_PATH  # + str(epoch + 1) + '_' + str(batch_idx + 1)
            student.save_weights(Ssave_path)
            SBacc = Sacc
            print(f'saving for SBacc {SBacc}, Spath:{Ssave_path}')