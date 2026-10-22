#!/usr/bin/env python3
"""Variational autoencoder"""
from tensorflow import keras

K = keras.ops if hasattr(keras, "ops") else keras.backend


class Sampling(keras.layers.Layer):
    """Samples the latent vector and adds the KL divergence loss"""

    def __init__(self, **kwargs):
        """Initializes the layer with a seeded random generator"""
        super().__init__(**kwargs)
        if hasattr(keras, "random"):
            self.seed_generator = keras.random.SeedGenerator(42)

    def call(self, inputs):
        """
        Applies the reparametrization trick.

        Args:
            inputs: list with the mean and the log variance tensors.

        Returns:
            The latent vector mu + exp(log_var / 2) * epsilon.
        """
        mu, log_var = inputs
        if hasattr(keras, "random"):
            eps = keras.random.normal(K.shape(mu), seed=self.seed_generator)
        else:
            eps = K.random_normal(K.shape(mu))
        kl = -0.5 * K.sum(
            1 + log_var - K.square(mu) - K.exp(log_var), axis=-1
        )
        self.add_loss(K.mean(kl))
        return mu + K.exp(log_var / 2) * eps


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims: integer, dimensions of the model input.
        hidden_layers: list with the number of nodes of each hidden layer
            in the encoder. The decoder uses the reversed list.
        latent_dims: integer, dimensions of the latent space.

    Returns:
        encoder, decoder, auto: the encoder (outputs the latent vector,
        the mean and the log variance), the decoder and the full
        autoencoder. The autoencoder is compiled with adam and binary
        cross-entropy, plus the KL divergence term.
    """
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation="relu")(x)
    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)
    z = Sampling()([mu, log_var])
    encoder = keras.Model(inputs, [z, mu, log_var])

    latent_inputs = keras.Input(shape=(latent_dims,))
    y = latent_inputs
    for units in reversed(hidden_layers):
        y = keras.layers.Dense(units, activation="relu")(y)
    outputs = keras.layers.Dense(input_dims, activation="sigmoid")(y)
    decoder = keras.Model(latent_inputs, outputs)

    auto = keras.Model(inputs, decoder(z))

    def loss(y_true, y_pred):
        """Binary cross-entropy summed over the input dimensions"""
        return keras.losses.binary_crossentropy(y_true, y_pred) * input_dims

    auto.compile(optimizer="adam", loss=loss)

    return encoder, decoder, auto
