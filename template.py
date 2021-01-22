import tensorflow as tf
import numpy as np


def my_op(a, b, c, g):
    with g.name_scope('MyOp') as scope:
        sum = tf.reduce_sum(a, name=scope)
        return sum


if __name__ == '__main__':
    g = tf.Graph()
    # with g.as_default():
    #     c = tf.constant(30.0)
    #     assert c.graph is g
    #     with g.name_scope('scope1'):
    #         with g.name_scope('scope2'):
    #             print(g.get_name_scope())
    a = tf.ones((2, 2))
    b = tf.constant(30.0)
    c = tf.constant(30.0)
    #
    # a = tf.convert_to_tensor(a, name='a')
    # b = tf.convert_to_tensor(b, name='b')
    # c = tf.convert_to_tensor(c, name='c')

    with g.as_default():
        sum = my_op(a, b, c, g)
        print(g.get_name_scope())

