import tensorflow as tf


class Dense(tf.Module):
    def __init__(self, num_inp_filters, num_out_filters, use_bias=True, name='dense'):
        super().__init__(name=name)
        self.num_inp_filters = num_inp_filters
        self.num_out_filters = num_out_filters
        self.use_bias = use_bias
        self.w = tf.Variable(tf.random.normal([self.num_inp_filters, self.num_out_filters]), name='w')
        self.w = tf.cast(self.w, tf.bfloat16)
        if self.use_bias:
            self.b = tf.Variable(tf.zeros([self.num_out_filters]), name='b')
            self.b = tf.cast(self.b, tf.bfloat16)

    def __call__(self, x):
        x = tf.linalg.matmul(x, self.w)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.b, name='bias_add')
        return x