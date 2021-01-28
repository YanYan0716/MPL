import tensorflow as tf
import tensorflow.keras as keras


import config


def LearningRate(initial_lr, num_warmup_steps=None, num_wait_steps=None):
    initial_lr = initial_lr * config.BATCH_SIZE / 256

    if config.LR_DECAY_TYPE == 'constant':
        lr = tf.constant(initial_lr, dtype=tf.float32)

    elif config.LR_DECAY_TYPE == 'exponential':
        lr = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=config.NUM_DECAY_STEPS,
            decay_rate=config.LR_DECAY_RATE,
        )

    elif config.LR_DECAY_TYPE == 'cosine':
        if num_wait_steps is None:
            lr = keras.experimental.CosineDecay(
                initial_learning_rate=initial_lr,
                decay_steps=config.NUM_DECAY_STEPS,
                alpha=0.0
            )
        else:
            lr = keras.experimental.CosineDecay(
                initial_learning_rate=initial_lr,
                decay_steps=config.NUM_DECAY_STEPS,
                alpha=0.0
            )
    else:
        raise ValueError(f'unknown lr_decay_type in config.py')

    return lr