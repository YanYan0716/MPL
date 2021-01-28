import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = 2

import tensorflow as tf

import config

if __name__ == '__main__':
    img = tf.random.normal([2, config.IMG_SIZE, config.IMG_SIZE, 3])
