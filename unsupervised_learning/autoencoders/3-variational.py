#!/usr/bin/env python3
"""Variational autoencoder"""
import tensorflow.keras as keras

K = keras.ops if hasattr(keras, "ops") else keras.backend


def sampling(args):
    """
    Samples the latent vector with the reparametrization trick.

    Args:
        args: list with the mean and the log variance tensors.

    Returns:
        The latent vector mu + exp(log_var / 2) * epsilon.
    """
    mu, log_var = args
    if hasattr(keras, "random"):
        eps = keras.random.normal(K.shape(mu))
    else:
        eps = K.random_normal(K.shape(mu))
    return mu + K.exp(log_var / 2) * eps


class VAELoss(keras.layers.Layer):
    """Adds the KL divergence and the reconstruction scale to the loss"""

    def __init__(self, input_dims, **kwargs):
        """Stores the number of input dimensions"""
        super().__init__(**kwargs)
        self.input_dims = input_dims

    def call(self, inputs):
        """
        Adds the extra loss terms and returns the reconstruction.

        The compiled binary cross-entropy gives the mean over the pixels.
        The extra (input_dims - 1) * bce makes the reconstruction loss a
        sum over the pixels, and the KL divergence is added on top.

        Args:
            inputs: list with the input, the reconstruction, the mean and
                the log variance tensors.

        Returns:
            The reconstruction tensor, unchanged.
        """
        x, x_hat, mu, log_var = inputs
        bce = K.mean(keras.losses.binary_crossentropy(x, x_hat))
        kl = -0.5 * K.sum(
            1 + log_var - K.square(mu) - K.exp(log_var), axis=-1
        )
        self.add_loss((self.input_dims - 1) * bce + K.mean(kl))
        return x_hat


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
        autoencoder, compiled with adam and binary cross-entropy.
    """
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation="relu")(x)
    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)
    z = keras.layers.Lambda(sampling)([mu, log_var])
    encoder = keras.Model(inputs, [z, mu, log_var])

    latent_inputs = keras.Input(shape=(latent_dims,))
    y = latent_inputs
    for units in reversed(hidden_layers):
        y = keras.layers.Dense(units, activation="relu")(y)
    outputs = keras.layers.Dense(input_dims, activation="sigmoid")(y)
    decoder = keras.Model(latent_inputs, outputs)

    z_out, mu_out, log_var_out = encoder(inputs)
    x_hat = decoder(z_out)
    x_hat = VAELoss(input_dims)([inputs, x_hat, mu_out, log_var_out])
    auto = keras.Model(inputs, x_hat)
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
