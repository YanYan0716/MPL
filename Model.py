import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from Conv2d import Conv2d
from WrnBlock import WrnBlock
from BatchNorm import BatchNorm
import config
from Dense import Dense


class Wrn28k(tf.Module):
    '''搭建最终网络结构'''

    def __init__(self, num_inp_filters, k=2, name='Wrn_28_2', training=True):
        super(Wrn28k, self).__init__(name=name)
        self.training = training
        self.s = [16, 135, 135 * 2, 135 * 4] if k == 135 else [16 * k, 16 * k, 32 * k, 64 * k]
        self.conv2d = Conv2d(
            num_inp_filters=num_inp_filters,
            filter_size=3,
            num_out_filters=self.s[0],
            stride=1,
        )
        self.wrn_block_1 = WrnBlock(num_inp_filters=self.s[1], num_out_filters=self.s[1], stride=1, training=training,
                                    name='wrn_block_1', activate_before_residual=True)
        self.wrn_block_2 = WrnBlock(num_inp_filters=self.s[1], num_out_filters=self.s[1], stride=1, training=training,
                                    name='wrn_block_2', activate_before_residual=True)
        self.wrn_block_3 = WrnBlock(num_inp_filters=self.s[1], num_out_filters=self.s[1], stride=1, training=training,
                                    name='wrn_block_3', activate_before_residual=True)
        self.wrn_block_4 = WrnBlock(num_inp_filters=self.s[1], num_out_filters=self.s[1], stride=1, training=training,
                                    name='wrn_block_4', activate_before_residual=True)

        self.wrn_block_5 = WrnBlock(num_inp_filters=self.s[1], num_out_filters=self.s[2], stride=2, training=training,
                                    name='wrn_block_5', activate_before_residual=False)
        self.wrn_block_6 = WrnBlock(num_inp_filters=self.s[2], num_out_filters=self.s[2], stride=1, training=training,
                                    name='wrn_block_6', activate_before_residual=False)
        self.wrn_block_7 = WrnBlock(num_inp_filters=self.s[2], num_out_filters=self.s[2], stride=1, training=training,
                                    name='wrn_block_7', activate_before_residual=False)
        self.wrn_block_8 = WrnBlock(num_inp_filters=self.s[2], num_out_filters=self.s[2], stride=1, training=training,
                                    name='wrn_block_8', activate_before_residual=False)

        self.wrn_block_9 = WrnBlock(num_inp_filters=self.s[2], num_out_filters=self.s[3], stride=2, training=training,
                                    name='wrn_block_9', activate_before_residual=False)
        self.wrn_block_10 = WrnBlock(num_inp_filters=self.s[3], num_out_filters=self.s[3], stride=1, training=training,
                                     name='wrn_block_10', activate_before_residual=False)
        self.wrn_block_11 = WrnBlock(num_inp_filters=self.s[3], num_out_filters=self.s[3], stride=1, training=training,
                                     name='wrn_block_11', activate_before_residual=False)
        self.wrn_block_12 = WrnBlock(num_inp_filters=self.s[3], num_out_filters=self.s[3], stride=1, training=training,
                                     name='wrn_block_12', activate_before_residual=False)

        self.bn = BatchNorm(size=self.s[3], training=self.training)
        self.dense = Dense(num_inp_filters=self.s[3], num_out_filters=config.NUM_CLASSES)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 32, 32, 3], dtype=config.DTYPE)])
    def __call__(self, x):
        self.conv2d.training = self.training
        self.wrn_block_1.training = self.training
        self.wrn_block_2.training = self.training
        self.wrn_block_3.training = self.training
        self.wrn_block_4.training = self.training
        self.wrn_block_5.training = self.training
        self.wrn_block_6.training = self.training
        self.wrn_block_7.training = self.training
        self.wrn_block_8.training = self.training
        self.wrn_block_9.training = self.training
        self.wrn_block_10.training = self.training
        self.wrn_block_11.training = self.training
        self.wrn_block_12.training = self.training
        self.bn.training = self.training
        self.dense.training = self.training

        x = self.conv2d(x)
        x = self.wrn_block_1(x)
        x = self.wrn_block_2(x)
        x = self.wrn_block_3(x)
        x = self.wrn_block_4(x)
        x = self.wrn_block_5(x)
        x = self.wrn_block_6(x)
        x = self.wrn_block_7(x)
        x = self.wrn_block_8(x)
        x = self.wrn_block_9(x)
        x = self.wrn_block_10(x)
        x = self.wrn_block_11(x)
        x = self.wrn_block_12(x)

        x = self.bn(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = tf.reduce_mean(x, axis=[1, 2], name='global_avg_pool')
        x = tf.nn.dropout(x, rate=config.DROPOUT_RATE)
        x = self.dense(x)
        x = tf.cast(x, dtype=config.DTYPE, name='logits')
        return x


if __name__ == '__main__':
    img = tf.random.normal([1, config.IMG_SIZE, config.IMG_SIZE, 3], dtype=config.DTYPE)
    model = Wrn28k(num_inp_filters=3, k=2, training=True)
    model.training = False

    # print(len(model.trainable_variables))
    output = model(x=img)
    # print(model.bn.moving_variance)
    # print(model.bn.moving_mean)
    # print(output.shape)

    # checkpoint_dir = './weights'
    # checkpoint_prefix = checkpoint_dir + '/ckpt'

    # checkpoint.model = model
    # status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # status.assert_consumed()
    # checkpoint.save(checkpoint_prefix)
    # print(model(img).shape)
    # tf.saved_model.save(model, './weights')
    # print('saving checkpoint ...')
