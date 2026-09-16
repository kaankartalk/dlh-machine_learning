#!/usr/bin/env python3

import tensorflow as tf
import matplotlib.pyplot as plt

flip_image = __import__('0-flip').flip_image

tf.compat.v1.enable_eager_execution()
doggies = tf.data.Dataset.list_files('./animals/doggies/*.jpg', seed=0)
for image in doggies.shuffle(10).take(1):
    flipped_image = flip_image(image)
    plt.imshow(flipped_image)
    plt.show()
