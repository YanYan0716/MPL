'''
reference: https://arxiv.org/pdf/1502.03167v3.pdf
'''
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

import config


def shared_weight(w, num_cores):
    del num_cores
    return w


@tf.function
def update(moving, normal):
    momentum = tf.cast(config.BATCH_NORM_DECAY, tf.float32)
    moving = momentum * moving + tf.cast(1.0 - momentum, tf.float32) * normal
    return moving


class BatchNorm(tf.Module):
    def __init__(self, size, training, name='BatchNorm'):
        super(BatchNorm, self).__init__(name=name)
        self.size = size
        self.training = training
        self.gamma = tf.Variable(initial_value=tf.ones([self.size], dtype=tf.float32), trainable=True, name='gamma')
        # self.gamma = shared_weight(w=self.gamma, num_cores=config.NUM_XLA_SHARDS)
        self.bate = tf.Variable(initial_value=tf.zeros([self.size], dtype=tf.float32), trainable=True, name='bate')
        # self.bate = shared_weight(w=self.bate, num_cores=config.NUM_XLA_SHARDS)
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

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)])
    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        if self.training:
            x, mean, variance = tf.compat.v1.nn.fused_batch_norm(
                x=x,
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
        else:
            x, _, _ = tf.compat.v1.nn.fused_batch_norm(
                x,
                scale=self.gamma,
                offset=self.bate,
                mean=self.moving_mean,
                variance=self.moving_variance,
                epsilon=config.BATCH_NORM_EPSILON,
                is_training=False
            )
        return x


def loss(input):
    value = tf.reduce_mean(input + 1)
    value = tf.expand_dims(value, axis=0)
    value = tf.expand_dims(value, axis=0)
    return value


if __name__ == '__main__':
    np.random.seed(1)
    img = np.random.random([1, 32, 32, 3])
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    opt = tf.keras.optimizers.SGD(learning_rate=0.001)

    # 使用自己定义的BN层  有两个变量
    model_m = BatchNorm(3, training=True)
    # print(len(model_m.trainable_variables))

    tf.saved_model.save(model_m, './weights')

    # for i in range(2):
    #     with tf.GradientTape() as tape:
    #         output_m = model_m(img)
    #         Loss = loss(output_m)
    #         print(model_m.moving_mean)
    #         print(model_m.moving_variance)
    #     grad = tape.gradient(Loss, model_m.trainable_variables)
    #     opt.apply_gradients(zip(grad, model_m.trainable_variables))
        # print(model_m.trainable_variables)
