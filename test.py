import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import config
from Model import Wrn28k

if __name__ == '__main__':
    img_file = './100.jpg'
    # 对图片的处理
    img = tf.io.read_file(img_file)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    # print(img[0][0][0])

    # 加载模型
    ck = tf.saved_model.load('./weights')
    print(ck(img)[0][0][0])
