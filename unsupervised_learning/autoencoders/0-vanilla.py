#!/usr/bin/env python3
"""Vanilla autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder.

    Args:
        input_dims: integer, dimensions of the model input.
        hidden_layers: list with the number of nodes of each hidden layer
            in the encoder. The decoder uses the reversed list.
        latent_dims: integer, dimensions of the latent space.

    Returns:
        encoder, decoder, auto: the encoder, the decoder and the full
        autoencoder. The autoencoder is compiled with adam and binary
        cross-entropy.
    """
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation="relu")(x)
    latent = keras.layers.Dense(latent_dims, activation="relu")(x)
    encoder = keras.Model(inputs, latent)

    latent_inputs = keras.Input(shape=(latent_dims,))
    y = latent_inputs
    for units in reversed(hidden_layers):
        y = keras.layers.Dense(units, activation="relu")(y)
    outputs = keras.layers.Dense(input_dims, activation="sigmoid")(y)
    decoder = keras.Model(latent_inputs, outputs)

    auto = keras.Model(inputs, decoder(encoder(inputs)))
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
