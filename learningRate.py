import tensorflow as tf
import tensorflow.keras as keras


import config


class LearningRate(object):
    def __init__(self, initial_lr, num_warmup_steps, num_wait_steps):
        if initial_lr is None:
            raise ValueError(f'initial_lr is error in learningRate file')
        if num_warmup_steps is None:
            raise ValueError(f'num_warmup_steps is error in learningRate file')
        if num_wait_steps is None:
            raise ValueError(f'num_wait_steps is error in learningRate file')

        initial_lr = initial_lr * config.BATCH_SIZE / 256
        self.initial_lr = initial_lr
        self.num_warmup_steps = num_warmup_steps
        self.num_wait_steps = num_wait_steps

        if config.LR_DECAY_TYPE == 'constant':
            self.lr = tf.constant(self.initial_lr, dtype=tf.float32)

        elif config.LR_DECAY_TYPE == 'exponential':
            self.lr = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.initial_lr,
                decay_steps=config.NUM_DECAY_STEPS,
                decay_rate=config.LR_DECAY_RATE,
            )

        elif config.LR_DECAY_TYPE == 'cosine':
            self.lr = keras.experimental.CosineDecay(
                initial_learning_rate=self.initial_lr,
                decay_steps=config.MAX_STEPS - self.num_wait_steps - self.num_warmup_steps,
                alpha=0.0
            )
        else:
            raise ValueError(f'unknown lr_decay_type in config.py')

    def __call__(self, global_step):
        if config.LR_DECAY_TYPE == 'constant':
            learn_rate = self.lr
        else:
            learn_rate = self.lr.__call__(global_step)
        r = (tf.cast(global_step+1), tf.float32) / tf.cast(self.num_warmup_steps, tf.float32)
        warmup_lr = self.initial_lr * r
        lr = tf.cond(
            tf.cast(global_step, tf.int32) < tf.cast(self.num_warmup_steps, tf.float32),
            lambda : warmup_lr,
            lambda : learn_rate,
        )
        lr = tf.cond(global_step < 0, lambda :tf.constant(0., tf.float32), lambda :lr)
        return lr



'''
def LearningRate(initial_lr, num_warmup_steps, num_wait_steps):
    if initial_lr is None:
        raise ValueError(f'initial_lr is error in learningRate file')
    if num_warmup_steps is None:
        raise ValueError(f'num_warmup_steps is error in learningRate file')
    if num_wait_steps is None:
        raise ValueError(f'num_wait_steps is error in learningRate file')

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
        lr = keras.experimental.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=config.MAX_STEPS - num_wait_steps - num_warmup_steps,
            alpha=0.0
        )
    else:
        raise ValueError(f'unknown lr_decay_type in config.py')

    return lr
'''