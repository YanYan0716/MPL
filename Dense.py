import tensorflow as tf

import config


def shared_weight(w, num_cores):
    del num_cores
    return w


class Dense(tf.Module):
    def __init__(self, num_inp_filters, num_out_filters, use_bias=True, name='dense', training=True):
        super(Dense, self).__init__(name=name)
        self.num_inp_filters = num_inp_filters
        self.num_out_filters = num_out_filters
        self.use_bias = use_bias
        self.training = training
        init_range = 1. / tf.math.sqrt(tf.cast(self.num_out_filters, config.DTYPE))
        self.w = tf.Variable(
            initial_value=tf.random_uniform_initializer(
                minval=-init_range,
                maxval=init_range,
            )(shape=[self.num_inp_filters, self.num_out_filters], dtype=config.DTYPE),
            # tf.random.uniform(
            #     shape=[self.num_inp_filters, self.num_out_filters],
            #     minval=-init_range,
            #     maxval=init_range,
            #     dtype=config.DTYPE,
            # ),
            name='w',
            trainable=True
        )
        # self.w = shared_weight(self.w, num_cores=config.NUM_XLA_SHARDS)
        if self.use_bias:
            self.b = tf.Variable(
                initial_value=tf.zeros_initializer()(shape=[self.num_out_filters], dtype=config.DTYPE),
                # tf.zeros([self.num_out_filters], dtype=config.DTYPE),
                name='b',
                trainable=True
            )
            # self.b = shared_weight(self.b, num_cores=config.NUM_XLA_SHARDS)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=config.DTYPE)])
    def __call__(self, x):
        # self.w.trainable = self.training
        x = tf.linalg.matmul(x, self.w)
        if self.use_bias:
            # self.b.trainable = self.training
            x = tf.nn.bias_add(x, self.b, name='bias_add')
        return x
