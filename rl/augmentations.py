"""Data augmentations based on tf's functions"""

import tensorflow as tf

from typing import Union


def tf_normalize(image):
    """Scale the given image in range [0.0, 1.0]"""
    image -= tf.reduce_min(image)
    image /= tf.reduce_max(image)
    return image


def tf_chance():
    """Use to get a single random number between 0 and 1"""
    return tf.random.uniform(shape=(1,), minval=0.0, maxval=1.0)


def tf_resize(image, size: Union[tuple, tf.TensorShape]):
    return tf.image.resize(image, size)


def tf_saturation(image, lower=0.5, upper=1.5):
    return tf.image.random_saturation(image, lower, upper)


def tf_contrast(image, lower=0.4, upper=1.6):
    return tf.image.random_contrast(image, lower, upper)


def tf_brightness(image, delta=0.75):
    return tf.image.random_brightness(image, max_delta=delta)


def tf_hue(image, delta=0.5):
    return tf.image.random_hue(image, max_delta=delta)


def tf_grayscale(rgb_image):
    return tf.image.rgb_to_grayscale(rgb_image)


def tf_rgb(gray_image):
    return tf.image.grayscale_to_rgb(gray_image)


def tf_repeat_channels(image, n=3):
    if len(image.shape) == 2:
        return tf.stack((image,) * n, axis=-1)

    return tf.concat((image,) * n, axis=-1)


def tf_gaussian_noise(image, amount=0.25, std=0.2):
    mask_select = tf.keras.backend.random_binomial(image.shape[:2], p=amount)
    mask_select = tf.stack((mask_select,) * 3, axis=-1)
    mask_noise = tf.random.normal(shape=image.shape, stddev=std)
    return image + (mask_select * mask_noise)


def tf_salt_and_pepper(image, amount=0.1, prob=0.5):
    # source: https://stackoverflow.com/questions/55653940/how-do-i-implement-salt-pepper-layer-in-keras
    mask_select = tf.keras.backend.random_binomial(image.shape[:2], p=amount / 10)
    mask_select = tf.stack((mask_select,) * 3, axis=-1)

    mask_noise = tf.keras.backend.random_binomial(image.shape[:2], p=prob)
    mask_noise = tf.stack((mask_noise,) * 3, axis=-1)
    return image * (1 - mask_select) + mask_noise * mask_select


def tf_gaussian_blur(image, size=5, std=0.25):
    # source: https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319
    gaussian_kernel = tf.random.normal(shape=(size, size, image.shape[-1], 1), mean=1.0, stddev=std)
    image = tf.nn.depthwise_conv2d([image], gaussian_kernel, [1, 1, 1, 1], padding='SAME',
                                   data_format='NHWC')[0]
    return image


def tf_median_blur(image, size=5):
    mean_kernel = tf.ones((size, size, image.shape[-1], 1))
    image = tf.nn.depthwise_conv2d([image], mean_kernel, [1, 1, 1, 1], padding='SAME',
                                   data_format='NHWC')[0]
    return image


def tf_cutout(image, size=5):
    cut_mask = tf.random.normal(shape=(size, size))
    cut_mask = tf.where(condition=cut_mask == tf.reduce_max(cut_mask), x=0.0, y=1.0)
    cut_mask = tf.stack((cut_mask,) * 3, axis=-1)
    cut_mask = tf.image.resize([cut_mask], size=image.shape[:2],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0]
    return image * cut_mask


def tf_coarse_dropout(image, size=25, amount=0.1):
    drop_mask = tf.keras.backend.random_binomial((size, size), p=1.0 - amount)
    drop_mask = tf.stack((drop_mask,) * 3, axis=-1)
    drop_mask = tf.image.resize([drop_mask], size=image.shape[:2],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0]
    return image * drop_mask
