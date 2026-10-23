#!/usr/bin/env python3
"""Generator and discriminator for a convolutional face GAN"""
from tensorflow import keras


def convolutional_GenDiscr():
    """
    Builds a convolutional generator and a convolutional discriminator.

    The generator maps a latent vector of shape (16,) to a (16, 16, 1)
    picture. The discriminator maps a (16, 16, 1) picture to a score
    in [-1, 1]. All activations are tanh.

    Returns:
        generator, discriminator: the two keras Models.
    """

    def get_generator():
        """Builds the generator model"""
        inputs = keras.Input(shape=(16,))
        x = keras.layers.Dense(2048, activation="tanh")(inputs)
        x = keras.layers.Reshape((2, 2, 512))(x)

        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(64, (3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(16, (3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(1, (3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation("tanh")(x)

        return keras.Model(inputs, outputs, name="generator")

    def get_discriminator():
        """Builds the discriminator model"""
        inputs = keras.Input(shape=(16, 16, 1))
        x = inputs
        for filters in [32, 64, 128, 256]:
            x = keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
            x = keras.layers.MaxPooling2D()(x)
            x = keras.layers.Activation("tanh")(x)
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1, activation="tanh")(x)

        return keras.Model(inputs, outputs, name="discriminator")

    return get_generator(), get_discriminator()
