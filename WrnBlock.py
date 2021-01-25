import tensorflow as tf


from BatchNorm import BatchNorm
from Conv2d import Conv2d


class WrnBlock(tf.Module):
    def __init__(self, num_inp_filters, num_out_filters, stride, training=True, name='wrn_block' ):
        super().__init__(name=name)
        self.num_inp_filters = num_inp_filters
        self.num_out_filters = num_out_filters
        self.stride = stride
        self.training = training
        self.batch_norm_1 = BatchNorm(
            size=self.num_inp_filters,
            training=self.training,
            name='bn_1'
        )
        self.conv2d_1 = Conv2d(
            num_inp_filters=self.num_inp_filters,
            filter_size=3,
            num_out_filters=self.num_out_filters,
            stride=self.stride,
            name='conv_3*3_1',
        )
        self.batch_norm_2 = BatchNorm(
            size=self.num_inp_filters,
            training=self.training,
            name='bn_2',
        )
        self.conv2d_2 = Conv2d(
            num_inp_filters=self.num_inp_filters,
            filter_size=3,
            num_out_filters=self.num_out_filters,
            stride=1,
            name='conv_3*3_2'
        )
        self.residual = Conv2d(
            num_inp_filters=self.num_inp_filters,
            filter_size=1,
            num_out_filters=self.num_out_filters,
            stride=self.stride
        )

    def __call__(self, x):
        residual_x = x
        x = self.batch_norm_1(x)
        if self.stride == 2 or self.num_out_filters != self.num_inp_filters:
            residual_x = x
        x = tf.nn.leaky_relu(x, alpha=0.2, name='Lrelu_1')
        x = self.conv2d_1(x)
        x = self.batch_norm_2(x)
        x = tf.nn.leaky_relu(x, alpha=0.2, name='Lrelu_2')
        x = self.conv2d_2(x)
        if self.stride == 2 or self.num_out_filters != self.num_inp_filters:
            residual_x = tf.nn.leaky_relu(residual_x, alpha=0.2, name='Lrelu_3')
            residual_x = self.residual(residual_x)
        x = x + residual_x
        return x


# def wrn_block(x, params, num_out_filters, stride, training=True, name='wrn_black'):
#     pass