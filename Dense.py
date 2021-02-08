import tensorflow as tf


import config


def shared_weight(w, num_cores):
    del num_cores
    return w


class Dense(tf.Module):
    def __init__(self, num_inp_filters, num_out_filters, use_bias=True, name='dense'):
        super(Dense, self).__init__(name=name)
        self.num_inp_filters = num_inp_filters
        self.num_out_filters = num_out_filters
        self.use_bias = use_bias
        init_range = 1. / tf.math.sqrt(self.num_out_filters)
        self.w = tf.Variable(
            tf.random.uniform(
                shape=[self.num_inp_filters, self.num_out_filters],
                minval=-init_range,
                maxval=init_range
            ),
            name='w'
        )
        self.w = shared_weight(self.w, num_cores=config.NUM_XLA_SHARDS)
        if self.use_bias:
            self.b = tf.Variable(tf.zeros([self.num_out_filters]), name='b')
            self.b = shared_weight(self.b, num_cores=config.NUM_XLA_SHARDS)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def __call__(self, x):
        x = tf.linalg.matmul(x, self.w)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.b, name='bias_add')
        return x

