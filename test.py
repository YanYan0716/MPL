import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pandas as pd

import config


def test(student):
    student.training = False
    # 准备数据
    df_label = pd.read_csv(config.TEST_FILE_PATH)
    file_paths = df_label['file_name'].values
    labels = df_label['label'].values

    # testing
    total_num = len(labels)/2
    corrent_num = 0
    for i in range(total_num):
        img_file = file_paths[i]
        label = int(labels[i])

        # 对图片的处理
        img = tf.io.read_file(img_file)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, axis=0)

        # 网络
        output = student(img)
        output = tf.nn.softmax(output)
        class_index = tf.squeeze(tf.math.argmax(output, axis=1))

        if class_index == label:
            corrent_num += 1
    accuracy = float(corrent_num) / float(total_num) * 100.
    student.training = True
    return accuracy


if __name__ == '__main__':
    # 加载模型
    student = tf.saved_model.load('./weights')

    # 准备数据
    df_label = pd.read_csv(config.TEST_FILE_PATH)
    file_paths = df_label['file_name'].values
    labels = df_label['label'].values

    # testing
    total_num = len(labels)
    corrent_num = 0
    for i in range(total_num):
        img_file = file_paths[i]
        label = int(labels[i])

        # 对图片的处理
        img = tf.io.read_file(img_file)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, axis=0)

        # 网络
        output = student(img)
        output = tf.nn.softmax(output)
        class_index = tf.squeeze(tf.math.argmax(output, axis=1))

        if class_index == label:
            corrent_num += 1
    accuracy = float(corrent_num) / float(total_num) * 100.
    print(f'test acc : %.3f' % {accuracy} + '%')
