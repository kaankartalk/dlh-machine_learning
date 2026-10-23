#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

Simple_GAN = __import__('0-simple_gan').Simple_GAN


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def spheric_generator(nb_points, dim):
    u = tf.random.normal(shape=(nb_points, dim))
    return u / tf.reshape(
        tf.sqrt(tf.reduce_sum(tf.square(u), axis=[1]) + 10**-8),
        [nb_points, 1])


def normal_generator(nb_points, dim):
    return tf.random.normal(shape=(nb_points, dim))


def uniform_generator(nb_points, dim):
    return tf.random.uniform(shape=(nb_points, dim))


def fully_connected_GenDiscr(gen_shape, real_examples, latent_type="normal"):
    if latent_type == "uniform":
        latent_generator = lambda k: uniform_generator(k, gen_shape[0])
    elif latent_type == "normal":
        latent_generator = lambda k: normal_generator(k, gen_shape[0])
    elif latent_type == "spheric":
        latent_generator = lambda k: spheric_generator(k, gen_shape[0])

    inputs = keras.Input(shape=(gen_shape[0],))
    hidden = keras.layers.Dense(gen_shape[1], activation='tanh')(inputs)
    for i in range(2, len(gen_shape) - 1):
        hidden = keras.layers.Dense(gen_shape[i], activation='tanh')(hidden)
    outputs = keras.layers.Dense(gen_shape[-1], activation='sigmoid')(hidden)
    generator = keras.Model(inputs, outputs, name="generator")

    inputs = keras.Input(shape=(gen_shape[-1],))
    hidden = keras.layers.Dense(gen_shape[-2], activation='tanh')(inputs)
    for i in range(2, len(gen_shape) - 1):
        hidden = keras.layers.Dense(
            gen_shape[-1 * i], activation='tanh')(hidden)
    outputs = keras.layers.Dense(1, activation='tanh')(hidden)
    discriminator = keras.Model(inputs, outputs, name="discriminator")

    return generator, discriminator, latent_generator


def example_fully_connected_GAN(chosen_type, real_examples, gen_shape,
                                epochs, batch_size=200, steps_per_epoch=250,
                                latent_type="normal", learning_rate=.005):
    generator, discriminator, latent_generator = fully_connected_GenDiscr(
        gen_shape, real_examples, latent_type=latent_type)
    G = chosen_type(generator, discriminator, latent_generator,
                    real_examples, learning_rate=learning_rate)
    G.compile()
    G.fit(real_examples, epochs=epochs, steps_per_epoch=steps_per_epoch,
          verbose=1)
    return G


circle_example = lambda k: spheric_generator(k, 2) * .3 + .5

set_seeds(0)
G = example_fully_connected_GAN(Simple_GAN, circle_example(1000), [1, 10, 10, 2],
                                16, steps_per_epoch=100, learning_rate=.001)
