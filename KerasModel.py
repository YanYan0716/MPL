import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np


class WrnBlock(layers.Layer):
    def __init__(self, num_inp_filters, num_out_filters, strides, trainable=True, name='wrn_block'):
        super(WrnBlock, self).__init__(name=name)
        self.num_out_filters = num_out_filters
        self.strides = strides
        self.trainable = trainable,
        self.bn_1 = layers.BatchNormalization()
        self.conv2d_1 = layers.Conv2D(
            filters=self.num_out_filters,
            kernel_size=3,
            strides=self.strides,
            padding='SAME',
            use_bias=False,
            data_format='NHWC',
            trainable=self.trainable,
            name='conv_3_1',
        )
        self.bn_2 = layers.BatchNormalization()
        self.conv2d_2 = layers.Conv2D(
            filters=self.num_out_filters,
            kernel_size=3,
            strides=self.strides,
            padding='SAME',
            use_bias=False,
            data_format='NHWC',
            trainable=self.trainable,
            name='conv_3_2',
        )


if __name__ == '__main__':
    pass