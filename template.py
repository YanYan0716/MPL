import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


import config


def u(moving, normal):
    decay = tf.cast(1.-config.BATCH_NORM_DECAY, tf.float32)
    return moving.assign_sub(decay*(moving-normal))

class MyBn(tf.Module):
    def __init__(self, size, training=True, name='name'):
        super().__init__(name=name)
        self.size = size
        self.training = training
        self.gamma = tf.Variable(initial_value=tf.ones([self.size]), trainable=True, name='gamma')
        # print(self.gamma)
        self.bate = tf.Variable(initial_value=tf.zeros([self.size]), trainable=True, name='bate')
        # print(self.bate)
        self.moving_mean = tf.Variable(
            initial_value=tf.zeros([self.size]),
            trainable=False,
            name='moving_mean'
        )
        # print(self.moving_mean)
        self.moving_variance = tf.Variable(
            initial_value=tf.ones([self.size]),
            trainable=False,
            name='moving_variance'
        )
        # print(self.moving_variance)

    @tf.function
    def __call__(self, x):
        mean, variance = tf.nn.moments(x, [0, 1, 2])
        x = tf.nn.batch_normalization(
            x, mean=mean, variance=variance, offset=self.beta, scale=self.gamma,
            variance_epsilon=config.BATCH_NORM_EPSILON
        )
        self.add_update()
        return x


if __name__ == '__main__':
    np.random.seed(1)
    img = np.random.random((1, 2, 2, 3))
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    # print(img)
    model = MyBn(size=16)
    output = model(img)