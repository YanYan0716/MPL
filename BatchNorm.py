import tensorflow as tf
from tensorflow.keras import layers


import config


def shared_weights(w, num_scores):
    del num_scores
    return w


def u(decay, moving, normal):
    diff = decay * (moving - normal)
    return tf.math.reduce_sum(moving, diff, )


def batch_norm(x, training, name='batch_norm'):
    size = x.shape[-1].value
    gamma = tf.Variable(initial_value=lambda: tf.constant(1.0, shape=[size]), trainable=True)
    beta = tf.Variable(initial_value=lambda: tf.constant(0.0, shape=[size]), trainable=True)
    moving_mean = tf.Variable(initial_value=lambda: tf.constant(0.0, dtype=[size]), trainable=False)
    moving_variance = tf.Variable(initial_value=lambda: tf.constant(value=1.0, shape=[size]), trainable=False)
    gamma = shared_weights(gamma, config.NUM_XLA_SHARDS)
    beta = shared_weights(beta, config.NUM_XLA_SHARDS)
    if not training:
        moving_mean = shared_weights(moving_mean, config.NUM_XLA_SHARDS)
        moving_variance = shared_weights(moving_variance, config.NUM_XLA_SHARDS)

    x = tf.cast(x, tf.float32)
    if training:
        mean, variance = tf.nn.moments(x, [0, 1, 2])
        x = tf.nn.batch_normalization(x=x, mean=mean, variance=variance, offset=beta, scale=gamma, variance_epsilon=config.BATCH_NORM_EPSILON)
        x = tf.cast(x, tf.bfloat16, name='batch_norm_recast')
        if isinstance(moving_mean, tf.Variable) and isinstance(moving_variance, tf.Variable):
            decay = tf.cast(1. - config.BATCH_NORM_DECAY, tf.float32)
            tf.Graph.add_to_collection()
        else:
            return x, mean, variance
    else:
        x, _, _ = tf.nn.fused_batch_norm(x, scale=gamma, offset=beta, mean=moving_mean, vari)
        x = tf.cast(x, tf.bfloat16)
        return x


if __name__ == '__main__':
    pass