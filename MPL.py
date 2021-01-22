import tensorflow as tf
from tensorflow.keras import layers


class BatchNorm(layers.Layer):
    def __init__(self, x, params, trainable=True, name='batch_norm'):
        super(BatchNorm, self).__init__()
        self.size = x.shape[-1].value
        self.gamma = tf.Variable(initial_value=tf.initializers.ones(),
                                 )


    def call(self, inputs):
        pass