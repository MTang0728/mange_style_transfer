import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image

def load_image(image_url, image_size=(256, 256), greyscale=False, preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    # if image is from an online source, cache image file locally.
    if 'http' in image_url:
        image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
    else:
        image_path = image_url
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = tf.io.decode_image(
        tf.io.read_file(image_path),
        channels=3, dtype=tf.float32)[tf.newaxis, ...]
    # if check to RGB grey
    if greyscale:
        img = tf.image.rgb_to_grayscale(img, name=None)
        img = tf.image.grayscale_to_rgb(img, name=None)
    # crop image and resize
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def show_n(images, titles=('',)):
    n = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w  * n, w))
    gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
    for i in range(n):
        plt.subplot(gs[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i] if len(titles) > i else '')
    plt.show()

content_image_url = './images/content/tokyo.jpg'
style_image_url = './images/style/makoko2.jpg'

# The content image size can be arbitrary.
output_image_size = 384
content_img_size = (output_image_size, output_image_size)
content_image = load_image(content_image_url, content_img_size, greyscale=False)

style_img_size = (256, 256)  # Recommended to keep it at 256.
style_image = load_image(style_image_url, style_img_size, greyscale=False)
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')


# Load TF-Hub module.
hub_handle = './model/magenta_arbitrary-image-stylization-v1-256_2'
hub_module = hub.load(hub_handle)

# Stylize content image with given style image.
outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]

# Visualize input images and the generated stylized image.
show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])