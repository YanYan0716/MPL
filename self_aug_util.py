import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

_MAX_LEVEL = 10


def _enhance_level_to_arg(level):
    return (tf.cast((level / _MAX_LEVEL) * 1.8 + 0.1, tf.float32),)


def _translate_level_to_arg(level, translate_const):
    level = tf.convert_to_tensor((level / _MAX_LEVEL) * float(translate_const), tf.float32)
    should_flip = tf.cast(
        tf.floor(tf.random.uniform([]) + 0.5),
        tf.bool)  # 得到的结果为True和False
    final_tensor = tf.cond(should_flip, lambda: level, lambda: -level)
    return final_tensor


def _rotate_level_to_arg(level):
    level = tf.convert_to_tensor((level / _MAX_LEVEL) * 30, tf.float32)
    should_filp = tf.cast(
        tf.floor(tf.random.uniform([]) + 0.5),
        tf.bool
    )
    final_tensor = tf.cond(should_filp, lambda: level, lambda: -level)
    return final_tensor


def _shear_level_to_arg(level):
    level = tf.convert_to_tensor((level / _MAX_LEVEL) * 0.3, tf.float32)
    should_filp = tf.cast(
        tf.floor(tf.random.uniform([]) + 0.5),
        tf.bool
    )
    final_tensor = tf.cond(should_filp, lambda: level, lambda: -level)
    return final_tensor


def level_to_arg(cutout_const, translate_const):
    '''
    将对image做变化的函数所用到的参数整理成字典形式
    :param cutout_const:
    :param translate_const:
    :return: type:dict
    '''
    no_arg = lambda level: ()
    posterize_arg = lambda level: tf.cast(
        tf.cast(level, tf.float32) / _MAX_LEVEL * 4,
        tf.float32
    )
    solarize_arg = lambda level: tf.cast(
        tf.cast(level, tf.float32) / _MAX_LEVEL * 256,
        tf.float32
    )
    solarize_add_arg = lambda level: tf.cast(
        tf.cast(level, tf.float32) / _MAX_LEVEL * 110,
        tf.float32
    )
    cutout_arg = lambda level: tf.cast(
        tf.cast(level, tf.float32) / _MAX_LEVEL * cutout_const,
        tf.float32
    )
    translate_arg = lambda level: _translate_level_to_arg(level, translate_const)

    args = {
        'Identity': no_arg,
        'AutoContrast': no_arg,
        'Equalize': no_arg,
        'Invert': no_arg,
        'Rotate': _rotate_level_to_arg,
        'Posterize': posterize_arg,
        'Solarize': solarize_arg,
        'SplarizeAdd': solarize_add_arg,
        'Color': _enhance_level_to_arg,
        'Contrast': _enhance_level_to_arg,
        'Brightness': _enhance_level_to_arg,
        'Sharpness': _enhance_level_to_arg,
        'ShearX': _shear_level_to_arg,
        'ShearY': _shear_level_to_arg,
        'Cutout': cutout_arg,
        'TranslateX': translate_arg,
        'TranslateY': translate_arg,
    }
    return args


if __name__ == '__main__':
    a = level_to_arg(4, 5)
    b = a['Rotate'](4)
    print(b)
