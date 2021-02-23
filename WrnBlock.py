import tensorflow as tf

from BatchNorm import BatchNorm
from Conv2d import Conv2d

import config

class WrnBlock(tf.Module):
    def __init__(self, num_inp_filters, num_out_filters, stride, training=True, name='wrn_block'):
        super(WrnBlock, self).__init__(name=name)
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
            name='conv_3_1',
            training=self.training
        )
        self.batch_norm_2 = BatchNorm(
            size=self.num_out_filters,
            training=self.training,
            name='bn_2',
        )
        self.conv2d_2 = Conv2d(
            num_inp_filters=self.num_out_filters,
            filter_size=3,
            num_out_filters=self.num_out_filters,
            stride=1,
            name='conv_3_2',
            training=self.training
        )
        if self.stride == 2 or self.num_out_filters != self.num_inp_filters:
            self.residual = Conv2d(
                num_inp_filters=self.num_inp_filters,
                filter_size=1,
                num_out_filters=self.num_out_filters,
                stride=self.stride,
                training=self.training
            )

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=config.DTYPE)])
    def __call__(self, x):
        self.batch_norm_1.training = self.training
        self.conv2d_1.training = self.training
        self.batch_norm_2.training = self.training
        self.conv2d_2.training = self.training

        if self.stride == 2 or self.num_out_filters != self.num_inp_filters:
            x = tf.nn.leaky_relu(self.batch_norm_1(x), alpha=0.2)
        else:
            x_ = tf.nn.leaky_relu(self.batch_norm_1(x), alpha=0.2)

        if self.stride == 2 or self.num_out_filters != self.num_inp_filters:
            x_ = tf.nn.leaky_relu(self.batch_norm_2(self.conv2d_1(x)), alpha=0.2)
        else:
            x_ = tf.nn.leaky_relu(self.batch_norm_2(self.conv2d_1(x_)), alpha=0.2)

        if config.DROPOUT_RATE > 0:
            x_ = tf.nn.dropout(x_, rate=config.DROPOUT_RATE)
        x_ = self.conv2d_2(x_)

        if self.stride == 2 or self.num_out_filters != self.num_inp_filters:
            x = self.residual(x)
            x = tf.math.add(x, x_)
        else:
            x = tf.math.add(x, x_)

        #     residual_x = x
        # x = tf.nn.leaky_relu(x, alpha=0.2, name='Lrelu_1')
        # x = self.conv2d_1(x)
        # x = self.batch_norm_2(x)
        # x = tf.nn.leaky_relu(x, alpha=0.2, name='Lrelu_2')
        # x = self.conv2d_2(x)
        # if self.stride == 2 or self.num_out_filters != self.num_inp_filters:
        #     residual_x = tf.nn.leaky_relu(residual_x, alpha=0.2, name='Lrelu_3')
        #     self.residual.training = self.training
        #     residual_x = self.residual(residual_x)
        # x = tf.math.add(x, residual_x)

        # residual_x = x
        # x = self.batch_norm_1(x)
        # if self.stride == 2 or self.num_out_filters != self.num_inp_filters:
        #     residual_x = x
        # x = tf.nn.leaky_relu(x, alpha=0.2, name='Lrelu_1')
        # x = self.conv2d_1(x)
        # x = self.batch_norm_2(x)
        # x = tf.nn.leaky_relu(x, alpha=0.2, name='Lrelu_2')
        # x = self.conv2d_2(x)
        # if self.stride == 2 or self.num_out_filters != self.num_inp_filters:
        #     residual_x = tf.nn.leaky_relu(residual_x, alpha=0.2, name='Lrelu_3')
        #     self.residual.training = self.training
        #     residual_x = self.residual(residual_x)
        # x = tf.math.add(x, residual_x)

        return x


if __name__ == '__main__':
    img = tf.random.normal([1, 32, 32, 3], dtype=config.DTYPE)
    model = WrnBlock(num_inp_filters=3, num_out_filters=32, stride=1, training=True, name='wrn_block_1')
    output = model(img)
    print(output.shape)
    # print(len(model.trainable_variables))

    # tf.saved_model.save(model, './weights')
