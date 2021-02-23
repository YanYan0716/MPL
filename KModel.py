from tensorflow.keras import layers


class WrnBlock(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, name='wrn_block', trainable=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.trainable = trainable
        self.name = name
        self.bn1 = layers.BatchNormalization(
            trainable=self.trainable,
            name=self.name+'_bn1',
        )
        self.conv1 = layers.Conv2D(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            trainable=self.trainable,
            name=self.name+'_conv1',
        )
        self.bn2 = layers.BatchNormalization(
            trainable=self.trainable,
            name=self.name+'_bn2'
        )
        self.conv2 = layers.Conv2D(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            trainable=self.trainable,
            name=self.name+'_conv2'
        )
        if self.stride == 2 or (self.in_channels != self.out_channels):
            self.residual = layers.Conv2D(
                filters=self.out_channels,
                kernel_size=1,
                strides=1,
                trainable=self.trainable,
                name=self.name+'_residule',
            )

    def call(self, inputs):
        x = self.bn1(inputs)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)
