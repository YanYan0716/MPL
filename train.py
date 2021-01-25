import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = 2

import tensorflow as tf


if __name__ == '__main__':
    img = tf.random.normal([1, 32, 32, 3])