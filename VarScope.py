'''
关于variable_scope
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    x =tf.Variable(tf.constant(3.0), trainable=True)
    q = tf.Variable(tf.constant(3.0), trainable=True)
    with tf.GradientTape() as tape_0:
        y = q * q
        w = 2 * q
        y = tf.stop_gradient(y)
        m = y * 2 + 3 * w

    with tf.GradientTape() as tape_1:
        y = x * x
        w = 2 * x
        y = tf.stop_gradient(y)
        z = y * 2 + 3 * w
    grad_1 = tape_1.gradient(z, x)
    print(grad_1)
    with tape_0:
        m = m*3
    grad_0 = tape_0.gradient(m, q)
    print(grad_0)
