'''
reference:
https://github.com/google-research/google-research/tree/1f1741a985a0f2e6264adae985bde664a7993bd2/flax_models/cifar/datasets
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


import auto_augment


if __name__ == '__main__':
    path = 'E:/Algorithm/cifar/train/1/ambulance_s_000101.png'
    img = tf.io.read_file(path)
    print(auto_augment)

