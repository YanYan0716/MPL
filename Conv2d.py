import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


import config


def shared_weight(w, num_cores):
    del num_cores
    return w


class Conv2d(tf.Module):
    def __init__(self, num_inp_filters, filter_size, num_out_filters, stride=1, use_bias=False, padding='SAME',
                 data_format='NHWC', name='conv2d', b=None):
        super().__init__(name=name)
        self.stride = stride
        self.padding = padding
        self.data_format = data_format
        self.use_bias = use_bias
        self.w = tf.Variable(
            initial_value=lambda: tf.random.normal(
                shape=[filter_size, filter_size, num_inp_filters, num_out_filters],
                mean=0.0,
                stddev=np.sqrt(2.0 / int(filter_size * filter_size * num_out_filters))
            ),
            trainable=True,
        )

        if self.use_bias:
            if b is None:
                self.b = tf.Variable(
                    initial_value=lambda :tf.constant(0., shape=[num_out_filters]),
                    trainable=True,
                    name='bias',
                )

    def __call__(self, x):
        x = tf.nn.conv2d(x, self.w, strides=[1, self.stride, self.stride, 1], padding=self.padding, data_format=self.data_format)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.b, name='bias_add')
        return x





# def Conv2d(x, filter_size, num_out_filters, stride=1, use_bias=False,
#            padding='SAME', data_format='NHWC', name='conv2d', u=None, b=None, g=None):
#     with g.name_scope(name) as scope:
#         num_inp_filters = x.shape[-1]
#         w = tf.Variable(
#             initial_value=lambda: tf.random.normal(
#                 shape=[filter_size, filter_size, num_inp_filters, num_out_filters],
#                 mean=0.0,
#                 stddev=np.sqrt(2.0 / int(filter_size * filter_size * num_out_filters))
#             ),
#             trainable=True,
#         )
#         w = shared_weight(w, config.NUM_XLA_SHARDS)
#         x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding, data_format=data_format, name='baseConv')
#
#
#         if use_bias:
#             if b is None:
#                 b = tf.Variable(
#                     initial_value=lambda :tf.constant(0., shape=[num_out_filters]),
#                     trainable=True,
#                     name='bias',
#                 )
#                 b = shared_weight(b, config.NUM_XLA_SHARDS)
#                 b = tf.cast(b, tf.bfloat16)
#             x = tf.nn.bias_add(x, b, name='bias_add')
#         return x