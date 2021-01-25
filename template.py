import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
import numpy as np


@tf.function
def my_op(a, b, c, g):
    with g.name_scope('MyOp') as scope:
        sum = tf.reduce_sum(a, name=scope)
        return sum


if __name__ == '__main__':
    g = tf.Graph()
    with g.as_default():
        with g.name_scope('scope1'):
            with g.name_scope('scope2'):
                print(g.get_name_scope())

    inputs = tf.ones((2, 2))
    with g.name_scope('my_layer') as scope:
        weights = tf.Variable(initial_value=tf.constant(2.0, shape=(2, 2)), name='weights', shape=(2, 2))
        g.add_to_collection('weights', weights)
        biases = tf.Variable(initial_value=tf.constant(0.0, shape=(2, 2)), name='bias', shape=(2, 2))
        affine = tf.matmul(inputs, weights) + biases
        output = tf.nn.relu(affine, name=scope)
        print(g.get_collection('weights'))
