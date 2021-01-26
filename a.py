import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import normal
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
            mean, variance = tf.nn.moments(x, [0, 1, 2])
            self.moving_variance.assign(self.moving_variance*config.BATCH_NORM_DECAY+(1.0-config.BATCH_NORM_DECAY)*variance, use_locking=True)
            self.moving_mean.assign(self.moving_mean * config.BATCH_NORM_DECAY + (1.0 - config.BATCH_NORM_DECAY) * mean, use_locking=True)
            # self.moving_variance = update(self.moving_variance, variance)
            # self.moving_mean = update(self.moving_mean, mean)
            # print(self.moving_mean, self.moving_variance)
            x = tf.nn.batch_normalization(
                x=x,
                mean=self.moving_mean,
                variance=self.moving_variance,
                offset=self.bate,
                scale=self.gamma,
                variance_epsilon=config.BATCH_NORM_EPSILON,
            )
            return x


def loss(input):
    value = tf.reduce_mean(input)
    value = tf.expand_dims(value, axis=0)
    value = tf.expand_dims(value, axis=0)
    return value


if __name__ == '__main__':
    np.random.seed(1)
    img = np.ones([1, 32, 32, 3])
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    # 使用keras中的BN层
    bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.99, trainable=True, renorm=False, fused=False )
    model_k = tf.keras.Sequential()
    model_k.add(tf.keras.Input(shape=(None, None, 3)))
    model_k.add(bn)
    # for j in range(len(model_k.layers[0].weights)):
    #     print(model_k.layers[0].weights[j].trainable, model_k.layers[0].weights[j].name)

    # print(model_k.trainable_weights)
    # print('------------')
    for i in range(1):
        with tf.GradientTape() as tape:
            print(img[0][0][0])
            output_k = model_k(img, training=True)
            print(model_k.layers[0].weights)
            # print(output_k[0][0][0])
            Loss = loss(output_k)
            # print(Loss, )
        grad = tape.gradient(Loss, model_k.trainable_weights)
        opt.apply_gradients(zip(grad, model_k.trainable_weights))

    print('-------------------------------')
    # # 使用自己定义的BN层
    model_m = BatchNorm(3, training=True)
    # for i in range(len(model_m.trainable_variables)):
    #     print(model_m.trainable_variables[i])

    for i in range(1):
        with tf.GradientTape() as tape:
            print(img[0][0][0])
            output_m = model_m(img)
            print(model_m.variables)
            # print(output_m[0][0][0])

            Loss = loss(output_m)
            # print(Loss)
        grad = tape.gradient(Loss, model_m.trainable_variables)
        opt.apply_gradients(zip(grad, model_m.trainable_variables))

'''https://arxiv.org/pdf/1502.03167v3.pdf'''