import os
from abc import ABC

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


import config


class BasicBlock(layers.Layer):
    def __init__(self, in_channels, out_channels, stride, dropout, name, trainable):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dropout = dropout
        # name = name
        self.trainable = trainable

        self.bn1 = layers.BatchNormalization(
            momentum=0.001,
            trainable=self.trainable,
            name=name+'_bn1'
        )
        self.relu1 = layers.LeakyReLU(alpha=0.2)
        self.conv1 = layers.Conv2D(
            filters=self.out_channels,
            kernel_size=3,
            strides=self.stride,
            padding='same',
            use_bias=False,
            kernel_initializer=keras.initializers.HeNormal(),
            kernel_regularizer=regularizers.l2(config.WEIGHT_DECAY),
            trainable=self.trainable,
            name=name+'_conv1',
        )
        self.bn2 = layers.BatchNormalization(
            momentum=0.001,
            trainable=self.trainable,
            name=name+'_bn2'
        )
        self.relu2 = layers.LeakyReLU(alpha=0.2)
        self.dropout = layers.Dropout(
            rate=self.dropout,
            trainable=self.trainable,
            name=name+'_dropout',
        )
        self.conv2 = layers.Conv2D(
            filters=self.out_channels,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer=keras.initializers.HeNormal(),
            kernel_regularizer=regularizers.l2(config.WEIGHT_DECAY),
            trainable=self.trainable,
            name=name+'_conv2',
        )
        if self.stride != 1 or self.in_channels != self.out_channels:
            self.short_cut = layers.Conv2D(
                filters=self.out_channels,
                kernel_size=1,
                strides=self.stride,
                padding='same',
                use_bias=False,
                kernel_initializer=keras.initializers.HeNormal(),
                kernel_regularizer=regularizers.l2(config.WEIGHT_DECAY),
                trainable=self.trainable,
                name=name+'_shortcut'
            )
        self.add = layers.Add(name=name+'_add')

    def call(self, inputs, **kwargs):
        out = self.bn1(inputs)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.stride != 1 or self.in_channels != self.out_channels:
            shortcut = self.short_cut(inputs)
        else:
            shortcut = out
        out = self.add([shortcut, out])
        return out


class WideResnet(keras.Model):
    def __init__(self, k=[16, 32, 64, 128], name='wider'):
        super(WideResnet, self).__init__(name=name)
        self.k = k
        self.dropout = config.DROPOUT

        self.conv1 = layers.Conv2D(
            filters=k[0],
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer=keras.initializers.HeNormal(),
            kernel_regularizer=regularizers.l2(config.WEIGHT_DECAY),
            trainable=self.trainable,
            name=name + '_conv1',
        )
        self.Basic1 = BasicBlock(in_channels=k[0], out_channels=k[1], stride=1, dropout=self.dropout, name=name+'_Basic1', trainable=True)
        self.Basic2 = BasicBlock(in_channels=k[1], out_channels=k[1], stride=1, dropout=self.dropout, name=name+'_Basic2', trainable=True)
        self.Basic3 = BasicBlock(in_channels=k[1], out_channels=k[1], stride=1, dropout=self.dropout, name=name+'_Basic3', trainable=True)
        self.Basic4 = BasicBlock(in_channels=k[1], out_channels=k[1], stride=1, dropout=self.dropout, name=name+'_Basic4', trainable=True)

        self.Basic5 = BasicBlock(in_channels=k[1], out_channels=k[2], stride=2, dropout=self.dropout, name=name+'_Basic5', trainable=True)
        self.Basic6 = BasicBlock(in_channels=k[2], out_channels=k[2], stride=1, dropout=self.dropout, name=name+'_Basic6', trainable=True)
        self.Basic7 = BasicBlock(in_channels=k[2], out_channels=k[2], stride=1, dropout=self.dropout, name=name+'_Basic7', trainable=True)
        self.Basic8 = BasicBlock(in_channels=k[2], out_channels=k[2], stride=1, dropout=self.dropout, name=name+'_Basic8', trainable=True)

        self.Basic9 = BasicBlock(in_channels=k[2], out_channels=k[3], stride=2, dropout=self.dropout, name=name+'_Basic9', trainable=True)
        self.Basic10 = BasicBlock(in_channels=k[3], out_channels=k[3], stride=1, dropout=self.dropout, name=name+'_Basic10', trainable=True)
        self.Basic11 = BasicBlock(in_channels=k[3], out_channels=k[3], stride=1, dropout=self.dropout, name=name+'_Basic11', trainable=True)
        self.Basic12 = BasicBlock(in_channels=k[3], out_channels=k[3], stride=1, dropout=self.dropout, name=name+'_Basic12', trainable=True)

        self.bn1 = layers.BatchNormalization(
            momentum=0.001,
            trainable=self.trainable,
            name=name+'_bn1'
        )
        self.relu1 = keras.activations.relu

        self.avgpool = layers.GlobalAveragePooling2D(name=name+'_avgpool')
        self.dense = layers.Dense(
            units=config.NUM_CLASS,
            kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1.),
            # activation='softmax',
            kernel_regularizer=regularizers.l2(config.WEIGHT_DECAY),
            name=name+'_dense',
        )

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.Basic1(x)
        x = self.Basic2(x)
        x = self.Basic3(x)
        x = self.Basic4(x)
        x = self.Basic5(x)
        x = self.Basic6(x)
        x = self.Basic7(x)
        x = self.Basic8(x)
        x = self.Basic9(x)
        x = self.Basic10(x)
        x = self.Basic11(x)
        x = self.Basic12(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.avgpool(x)
        out = self.dense(x)
        return out

    def model(self):
        input = keras.Input(shape=(32, 32, 3), dtype=tf.float32)
        return keras.Model(inputs=input, outputs=self.call(input))


if __name__ == '__main__':
    img = tf.random.normal([1, 32, 32, 3])
    model = WideResnet().model()
    model.summary()