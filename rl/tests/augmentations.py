"""test about image augmentations"""

import cv2
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from rl import utils
from rl import augmentations as aug


def tf_augmentations(image, size=(150, 200)):
    image = image / np.max(image)

    # resize
    image_r = tf.image.resize(image, size)

    # saturation
    image_s = tf.image.random_saturation(image_r, lower=0.5, upper=1.5)

    # brightness
    image_b = tf.image.random_brightness(image_r, max_delta=0.75)

    # contrast
    image_c = tf.image.random_contrast(image_r, lower=0.4, upper=1.6)

    # hue
    image_hue = tf.image.random_hue(image_r, max_delta=0.5)

    # grayscale
    image_g = tf.image.rgb_to_grayscale(image_r)
    image_g = utils.depth_concat([image_g, image_g, image_g])

    # noise gaussian
    image_gauss = aug.tf_gaussian_noise(image_r, amount=0.25, std=0.2)

    # salt-and-pepper noise
    image_salt = aug.tf_salt_and_pepper(image_r, amount=0.1, prob=0.5)

    # gaussian blur
    image_gb = aug.tf_gaussian_blur(image_r, size=5, std=0.25)

    # median blur
    image_mb = aug.tf_median_blur(image_r, size=5)

    # cutout
    image_cut = aug.tf_cutout(image, size=5)

    # coarse dropout
    image_drop = aug.tf_coarse_dropout(image_r, size=25, amount=0.1)

    return [image_r, image_s, image_b, image_c, image_g, image_hue, image_gauss, image_salt, image_gb,
            image_mb, image_cut, image_drop]


def test_aug_with_dataset(batch_size=25):
    @tf.function
    def augmentations(image):
        image = tf.image.random_contrast(image, lower=0.4, upper=1.6)
        image = tf.image.random_hue(image, max_delta=0.5)

        if tf.random.uniform((1,), 0.0, 1.0) <= 0.5:
            image = aug.tf_gaussian_blur(image, size=7, std=0.25)

        if tf.random.uniform((1,), 0.0, 1.0) <= 0.75:
            image = aug.tf_normalize(image)
            image = aug.tf_salt_and_pepper(image, amount=0.1, prob=0.5)

        if tf.random.uniform((1,), 0.0, 1.0) <= 0.3:
            image = aug.tf_coarse_dropout(image, size=35)

        return aug.tf_normalize(image)

    @tf.function
    def augmentations2(image):
        image = aug.tf_normalize(image)

        # contrast, tone, saturation, brightness
        if aug.tf_chance() > 0.5:
            image = aug.tf_saturation(image)

        if aug.tf_chance() > 0.5:
            image = aug.tf_contrast(image, lower=0.5, upper=1.5)

        if aug.tf_chance() > 0.5:
            image = aug.tf_hue(image)

        if aug.tf_chance() > 0.5:
            image = aug.tf_brightness(image, delta=0.5)

        # blur
        if aug.tf_chance() < 0.33:
            image = aug.tf_gaussian_blur(image, size=5)

        # noise
        if aug.tf_chance() < 0.2:
            image = aug.tf_salt_and_pepper(image, amount=0.1)

        if aug.tf_chance() < 0.33:
            image = aug.tf_gaussian_noise(image, amount=0.15, std=0.15)

        image = aug.tf_normalize(image)

        # cutout & dropout
        if aug.tf_chance() < 0.15:
            image = aug.tf_cutout(image, size=6)

        if aug.tf_chance() < 0.10:
            image = aug.tf_coarse_dropout(image, size=49, amount=0.1)

        return image

    trace_path = 'traces/test-preprocess/trace-0-20200708-103027.npz'
    images = np.load(trace_path)['state_image']
    dataset = tf.data.Dataset.from_tensor_slices(images).map(augmentations2).batch(batch_size)

    for batch in dataset:
        utils.plot_images(list(batch))


if __name__ == '__main__':
    # ia.seed(1)

    # image = cv2.imread('carla.png')
    # image = cv2.resize(image, (200, 150))
    #
    # print(image.shape)
    # images = [image] * 4
    #
    # seq = utils.AugmentationPipeline(strength=0.75)
    # images_aug = seq.augment(images)
    # utils.plot_images(images_aug)

    # utils.plot_images(tf_augmentations(image))

    test_aug_with_dataset()
    pass
