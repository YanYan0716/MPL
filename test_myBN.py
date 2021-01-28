'''
测试keras中定义的BN层和自己定义的BN层
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import config


@tf.function
def update(moving, normal):
    momentum = tf.cast(config.BATCH_NORM_DECAY, tf.float32)
    moving = momentum * moving + tf.cast(1.0 - momentum, tf.float32) * normal
    return moving


class BatchNorm(tf.Module):
    def __init__(self, size, training, name='BatchNorm'):
        super().__init__(name=name)
        self.size = size
        self.training = training
        self.gamma = tf.Variable(initial_value=tf.ones([self.size], dtype=tf.float32), trainable=True, name='gamma')
        self.bate = tf.Variable(initial_value=tf.zeros([self.size], dtype=tf.float32), trainable=True, name='bate')
        self.moving_mean = tf.Variable(
            initial_value=tf.zeros([self.size], dtype=tf.float32),
            trainable=False,
            name='moving_mean'
        )
        self.moving_variance = tf.Variable(
            initial_value=tf.ones([self.size], dtype=tf.float32),
            trainable=False,
            name='moving_variance'
        )

    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        if self.training:
            x, mean, variance = tf.compat.v1.nn.fused_batch_norm(
                x=x,
                # mean=self.moving_mean,
                # variance=self.moving_variance,
                offset=self.bate,
                scale=self.gamma,
                epsilon=config.BATCH_NORM_EPSILON,
                is_training=True,
            )
            self.moving_variance.assign(
                self.moving_variance * config.BATCH_NORM_DECAY + (1.0 - config.BATCH_NORM_DECAY) * variance,
                use_locking=True)
            self.moving_mean.assign(self.moving_mean * config.BATCH_NORM_DECAY + (1.0 - config.BATCH_NORM_DECAY) * mean,
                                    use_locking=True)
            return x


def loss(inp):
    value = tf.reduce_mean(inp + 1)
    value = tf.expand_dims(value, axis=0)
    value = tf.expand_dims(value, axis=0)
    return value


if __name__ == '__main__':
    # np.random.seed(1)
    img = np.ones([1, 32, 32, 3])
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    opt = tf.keras.optimizers.SGD(learning_rate=0.001)

    # 使用keras中的BN层
    bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.99, trainable=True, )
    model_k = tf.keras.Sequential()
    model_k.add(tf.keras.Input(shape=(None, None, 3)))
    model_k.add(bn)

    for i in range(3):
        with tf.GradientTape() as tape:
            output_k = model_k(img, training=True)
            Loss = loss(output_k)
        grad = tape.gradient(Loss, model_k.trainable_weights)
        opt.apply_gradients(zip(grad, model_k.trainable_weights))
        print(Loss)
        print(model_k.layers[0].weights[2])

    print('-----------------')
    # 使用自己定义的BN层
    model_m = BatchNorm(3, training=True)
    # print(model_m.trainable_variables)
    for i in range(3):
        with tf.GradientTape() as tape:
            output_m = model_m(img)
            Loss = loss(output_m)
        grad = tape.gradient(Loss, model_m.trainable_variables)
        opt.apply_gradients(zip(grad, model_m.trainable_variables))
        print(Loss)
        print(model_m.moving_mean)

'''https://arxiv.org/pdf/1502.03167v3.pdf'''
