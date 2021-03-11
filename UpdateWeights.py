import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from copy import deepcopy
from tensorflow.keras import regularizers


import config


class A(keras.Model):
    def __init__(self, k=[16, 32, 64, 128], name='wider'):
        super(A, self).__init__(name=name)
        self.k = k
        self.dense = layers.Dense(
            units=10,
        )

    def call(self, inputs, training=None, mask=None):
        out = self.dense(inputs)
        return out

    def model(self):
        input = keras.Input(shape=(32, 32, 1), dtype=tf.float32)
        return keras.Model(inputs=input, outputs=self.call(input))


if __name__ == '__main__':
    img = tf.random.normal([1, 32, 32, 1])
    model = A().model()
    print(model.weights)
    model_A = deepcopy(model.weights)
    # print(model_A)
    with tf.GradientTape() as s_tape:
        logits = model(img)
        loss = tf.reduce_mean(logits)
    # 反向传播，更新参数-------
    TeaOptim = keras.optimizers.SGD(
        learning_rate=0.1,
        momentum=0.9,
        nesterov=True,
    )
    GStud_unlabel = s_tape.gradient(loss, model.trainable_variables)
    TeaOptim.apply_gradients(zip(GStud_unlabel, model.trainable_variables))
    print('1: ------------------')
    print(model.weights)
    print(model_A)
    # print(len(model_A))
    print('2: ----------------')
    for i in range(len(model_A)):
        model.weights[i] = model.weights[i].assign(model.weights[i]*0.005+model_A[i]*0.995)
    print(model.weights)
    print(model_A)
    print('3: --------------')
    model_A = deepcopy(model.weights)
    print(model_A)