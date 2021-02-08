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
        super(Conv2d, self).__init__(name=name)
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
        self.w = shared_weight(w=self.w, num_cores=config.NUM_XLA_SHARDS)
        if self.use_bias:
            if b is None:
                self.b = tf.Variable(
                    initial_value=lambda: tf.constant(0., shape=[num_out_filters]),
                    trainable=True,
                    name='bias',
                )
                self.b = shared_weight(w=self.b, num_cores=config.NUM_XLA_SHARDS)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)])
    def __call__(self, x):
        x = tf.nn.conv2d(x, self.w, strides=[1, self.stride, self.stride, 1], padding=self.padding,
                         data_format=self.data_format)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.b, name='bias_add')
        return x


if __name__ == '__main__':
    # 使用自己定义的Conv2d层  use_bias=True有两个变量, 否则有1个
    model = Conv2d(
            num_inp_filters=3,
            filter_size=1,
            num_out_filters=4,
            stride=1,
            use_bias=False
        )

    tf.saved_model.save(model, './weights')