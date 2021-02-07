import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt
import math
import tensorflow_addons.image as image_ops

import config


def autocontrast(image):
    lo = tf.cast(tf.reduce_min(image, axis=[0, 1]), tf.float32)
    hi = tf.cast(tf.reduce_max(image, axis=[0, 1]), tf.float32)
    scale = tf.math.divide(255.0, (hi - lo))
    offset = tf.math.multiply(-lo, scale)
    image = tf.math.add(
        tf.math.multiply(tf.cast(image, tf.float32), scale),
        offset
    )
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, tf.uint8)
    return image


def equalize(image):
    # image = tf.cast(image, tf.int32)
    # channel = tf.shape(image)[-1]
    # for i in range(channel):
    #     im = tf.cast(image[:, :, i], tf.int32)
    #     histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)
    #     nonzero = tf.where(tf.not_equal(histo, 0))
    #     nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    #     step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255
    #     print(step)
    #     if step == 0:
    #         pass
    #     else:
    #         lut = (tf.cumsum(histo) + (step // 2)) // step
    #         lut = tf.concat([[0], lut[:-1]], 0)
    #         lut = tf.clip_by_value(lut, 0, 255)
    #         # print(lut)
    #         image[:, :, i] = tf.gather(lut, image[:, :, i])
    #         # image[:, :, i] = im
    #     # image[:, :, i] = im

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(tf.equal(step, 0),
                         lambda: im,
                         lambda: tf.gather(build_lut(histo, step), im))
        return tf.cast(result, tf.uint8)

    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)

    return image


def invert(image):
    image = 255 - image
    return image


def rotate(image):
    level = tf.convert_to_tensor((config.AUGMENT_MAGNITUDE / config._MAX_LEVEL) * 30, tf.float32)
    should_filp = tf.cast(
        tf.floor(tf.random.uniform([]) + 0.5),
        tf.bool
    )
    degree = tf.cond(should_filp, lambda: level, lambda: -level)
    degree_to_radians = tf.convert_to_tensor(math.pi / 180., tf.float32)
    radians = tf.math.multiply(degree, degree_to_radians)
    new_imgsize = tf.cast(tf.math.abs(tf.divide(config.IMG_SIZE, radians)), tf.int32)
    image = tf.image.resize(image, (new_imgsize, new_imgsize))
    image = image_ops.rotate(image, radians, fill_mode='constant')
    image = tf.image.resize_with_crop_or_pad(image, config.IMG_SIZE, config.IMG_SIZE)
    image = tf.cast(image, tf.uint8)
    return image


def posterize(image):
    bit = tf.cast(tf.cast(config.AUGMENT_MAGNITUDE, tf.float32) / config._MAX_LEVEL * 4, tf.float32)
    shift = tf.cast(8 - bit, image.dtype)
    image = tf.bitwise.right_shift(image, shift)
    image = tf.bitwise.left_shift(image, shift)
    return image


def solarize_arg(image):
    threahold = tf.cast(tf.cast(config.AUGMENT_MAGNITUDE, tf.float32) / config._MAX_LEVEL * 22, tf.float32)
    threahold = tf.cast(threahold, image.dtype)
    image = tf.where(image < threahold, image, 255 - image)
    return image


def solarize_add(image, threahold=128):
    addition = tf.cast(tf.cast(config.AUGMENT_MAGNITUDE, tf.float32) / config._MAX_LEVEL * 2, tf.int32)
    threahold = tf.cast(threahold, image.dtype)
    image = tf.add(tf.cast(image, tf.int32), addition)
    image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
    image = tf.where(image < threahold, image, 255 - image)
    return image


def color(image, degenetate = None):
    if degenetate is None:
        degenerate  = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))

    factor = tf.cast((config.AUGMENT_MAGNITUDE / config._MAX_LEVEL) * 1.8 + 0.1, tf.float32)

    def _blend():
        degen = tf.image.convert_image_dtype(degenerate, tf.float32)
        img = tf.image.convert_image_dtype(image, tf.float32)
        output = degen + factor * (img - degen)
        output = tf.where(
            tf.logical_and(tf.less(0., factor), tf.less(factor, 1.)),
            x=output,
            y=tf.clip_by_value(output, 0., 255.)
        )
        return tf.image.convert_image_dtype(output, tf.uint8)

    pred_fn_pairs = [
        (tf.equal(factor, 0.), lambda: degenerate),
        (tf.equal(factor, 1.), lambda: image),
    ]
    image = tf.case(
        pred_fn_pairs=pred_fn_pairs,
        default=_blend,
        exclusive=True,
        strict=True,
    )
    return image


def contrast(image):
    degenerate = tf.image.rgb_to_grayscale(image)
    degenerate = tf.cast(degenerate, tf.int32)

    hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
    mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.
    degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
    degenerate = tf.clip_by_value(degenerate, 0., 255.)
    degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))

    factor = tf.cast((config.AUGMENT_MAGNITUDE / config._MAX_LEVEL) * 0.6 + 0.1, tf.float32)

    def _blend():
        degen = tf.image.convert_image_dtype(degenerate, tf.float32)
        img = tf.image.convert_image_dtype(image, tf.float32)
        output = degen + factor * (img - degen)
        output = tf.where(
            tf.logical_and(tf.less(0., factor), tf.less(factor, 1.)),
            x=output,
            y=tf.clip_by_value(output, 0., 255.)
        )
        return tf.image.convert_image_dtype(output, tf.uint8)

    pred_fn_pairs = [
        (tf.equal(factor, 0.), lambda: degenerate),
        (tf.equal(factor, 1.), lambda: image),
    ]
    image = tf.case(
        pred_fn_pairs=pred_fn_pairs,
        default=_blend,
        exclusive=True,
        strict=True,
    )
    return image


def brightness(image):
    image = tf.image.adjust_brightness(image, 0.25)
    return image


def sharpness(image):
    factor = tf.cast((config.AUGMENT_MAGNITUDE / config._MAX_LEVEL) * 1.6 + 0.1, tf.float32)
    image = tf.cast(image, tf.float32)
    image = image_ops.sharpness(image, factor)
    image = tf.cast(image, tf.uint8)
    return image


def shear_x(image):
    level = tf.convert_to_tensor((config.AUGMENT_MAGNITUDE / config._MAX_LEVEL) * 0.2, tf.float32)
    should_filp = tf.cast(
        tf.floor(tf.random.uniform([]) + 0.5),
        tf.bool
    )
    level = tf.cond(should_filp, lambda: level, lambda: -level)

    new_size = tf.cast(config.IMG_SIZE*1.2, dtype=tf.int32)
    image = tf.image.resize(image, (new_size, new_size))
    image = image_ops.shear_x(
        image,
        level,
        replace=tf.convert_to_tensor(config.REPLACE_COLOR, image.dtype)
    )
    image = tf.image.resize_with_crop_or_pad(image, config.IMG_SIZE, config.IMG_SIZE)
    image = tf.cast(image, tf.uint8)
    return image


def shear_y(image):
    level = tf.convert_to_tensor((config.AUGMENT_MAGNITUDE / config._MAX_LEVEL) * 0.2, tf.float32)
    should_filp = tf.cast(
        tf.floor(tf.random.uniform([]) + 0.5),
        tf.bool
    )
    level = tf.cond(should_filp, lambda: level, lambda: -level)

    new_size = tf.cast(config.IMG_SIZE*1.1, dtype=tf.int32)
    image = tf.image.resize(image, (new_size, new_size))
    image = image_ops.shear_y(
        image,
        level,
        replace=tf.convert_to_tensor(config.REPLACE_COLOR, image.dtype)
    )
    image = tf.image.resize_with_crop_or_pad(image, config.IMG_SIZE, config.IMG_SIZE)
    image = tf.cast(image, tf.uint8)
    return image


def translate_x(image):
    level = tf.convert_to_tensor(
        (config.AUGMENT_MAGNITUDE / config._MAX_LEVEL) * float(config.TRANSLATE_CONST), tf.float32)
    should_flip = tf.cast(
        tf.floor(tf.random.uniform([]) + 0.5),
        tf.bool)  # 得到的结果为True和False
    pixels = tf.cond(should_flip, lambda: level, lambda: -level)
    image = image_ops.translate(image, [-pixels, 0])
    return image


def translate_y(image):
    level = tf.convert_to_tensor(
        (config.AUGMENT_MAGNITUDE / config._MAX_LEVEL) * float(config.TRANSLATE_CONST), tf.float32)
    should_flip = tf.cast(
        tf.floor(tf.random.uniform([]) + 0.5),
        tf.bool)  # 得到的结果为True和False
    pixels = tf.cond(should_flip, lambda: level, lambda: -level)
    image = image_ops.translate(image, [0, -pixels])
    return image


def cutout(image):
    pad_size = tf.cast(
        tf.cast(config.AUGMENT_MAGNITUDE, tf.float32) / config._MAX_LEVEL * config.CUTOUT_CONST,
        tf.int32
    )
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    # Samples the center location in the image where the zero mask is applied.
    cutout_center_height = tf.random.uniform(
        shape=[], minval=0, maxval=image_height,
        dtype=tf.int32)

    cutout_center_width = tf.random.uniform(
        shape=[], minval=0, maxval=image_width,
        dtype=tf.int32)

    lower_pad = tf.maximum(0, cutout_center_height - pad_size)
    upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = tf.maximum(0, cutout_center_width - pad_size)
    right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [image_height - (lower_pad + upper_pad),
                    image_width - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=image.dtype),
        padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])
    image = tf.where(
        tf.equal(mask, 0),
        tf.ones_like(image, dtype=image.dtype) * config.REPLACE_COLOR,
        image)
    return image


def identity(image):
    return tf.identity(image)


if __name__ == '__main__':
    img_file = './100.jpg'
    # 加载图片
    img = tf.io.read_file(img_file)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = tf.cast(img, tf.uint8)
    # 处理图片
    img = identity(img)

    plt.imshow(img.numpy())
    plt.show()
