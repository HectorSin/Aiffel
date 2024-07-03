# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from base_code import get_clip_box, mixer
from data import num_classes
import random


# cutmix
def cutmix(image, label, prob = 1.0, batch_size = 16, img_size = 224):
    mixed_imgs = []
    mixed_labels = []

    for i in range(batch_size):
        image_a = image[i]
        label_a = label[i]
        j = tf.cast(tf.random.uniform([], 0, batch_size), tf.int32)
        image_b = image[j]
        label_b = label[j]
        x_min, y_min, x_max, y_max = get_clip_box(image_a, image_b)
        mixed_img, mixed_label = mixer(image_a, image_b, label_a, label_b, x_min, y_min, x_max, y_max)
        mixed_imgs.append(mixed_img)
        mixed_labels.append(mixed_label)

    mixed_imgs = tf.reshape(tf.stack(mixed_imgs), (batch_size, img_size, img_size, 3))
    mixed_labels = tf.reshape(tf.stack(mixed_labels), (batch_size, num_classes))
    return mixed_imgs, mixed_labels
"""
# augment
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.2, 0.5)
    image = tf.image.random_hue(image, 0.2)
    image = tf.image.random_saturation(image, 0.2, 0.5)
    image = tf.clip_by_value(image, 0, 1)
    return image, label
"""
    
# augment with randomness
def augment(image, label):
    if random.choice([True, False]):
        image = tf.image.random_flip_left_right(image)
    if random.choice([True, False]):
        image = tf.image.random_flip_up_down(image)
    if random.choice([True, False]):
        brightness_factor = random.uniform(0, 0.2)
        image = tf.image.random_brightness(image, brightness_factor)
    if random.choice([True, False]):
        contrast_lower = random.uniform(0.2, 0.5)
        contrast_upper = random.uniform(contrast_lower, 0.5)
        image = tf.image.random_contrast(image, contrast_lower, contrast_upper)
    if random.choice([True, False]):
        hue_factor = random.uniform(0, 0.2)
        image = tf.image.random_hue(image, hue_factor)
    if random.choice([True, False]):
        saturation_lower = random.uniform(0.2, 0.5)
        saturation_upper = random.uniform(saturation_lower, 0.5)  # Ensure upper is always >= lower
        image = tf.image.random_saturation(image, saturation_lower, saturation_upper)
    image = tf.clip_by_value(image, 0, 1)
    return image, label

