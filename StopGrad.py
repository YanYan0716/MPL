'''
重点了解 tf.stop_gradient
'''
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

if __name__ == '__main__':
    '''
    分析：对于以下方程可写为： z=2*(x^2) + (3* 2*w)
    正常情况下的导数： x=3 --> grad=18
    加入tf.stop_gradient(y)后阻断了y部分的导数计算，仅计算w部分的导数
    修改后的导数：  x=3 --> grad=6
    ** 注意在用tf.stop_gradient(y)时要对变量重新赋值， 即line 23
    '''
    x = tf.constant(3.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = x * x
        w = 2 * x
        y = tf.stop_gradient(y)
        z = y * 2 + 3 * w
    grad = tape.gradient(z, x)
    print(grad)
