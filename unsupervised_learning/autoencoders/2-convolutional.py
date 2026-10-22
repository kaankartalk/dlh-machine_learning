#!/usr/bin/env python3
"""Convolutional autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims: tuple of integers, dimensions of the model input.
        filters: list with the number of filters of each convolutional
            layer in the encoder. The decoder uses the reversed list.
        latent_dims: tuple of integers, dimensions of the latent space.

    Returns:
        encoder, decoder, auto: the encoder, the decoder and the full
        autoencoder. The autoencoder is compiled with adam and binary
        cross-entropy.
    """
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3), padding="same", activation="relu")(x)
        x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    encoder = keras.Model(inputs, x)

    latent_inputs = keras.Input(shape=latent_dims)
    y = latent_inputs
    for f in reversed(filters[1:]):
        y = keras.layers.Conv2D(
            f, (3, 3), padding="same", activation="relu")(y)
        y = keras.layers.UpSampling2D((2, 2))(y)
    y = keras.layers.Conv2D(
        filters[0], (3, 3), padding="valid", activation="relu")(y)
    y = keras.layers.UpSampling2D((2, 2))(y)
    outputs = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding="same", activation="sigmoid")(y)
    decoder = keras.Model(latent_inputs, outputs)

    auto = keras.Model(inputs, decoder(encoder(inputs)))
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
