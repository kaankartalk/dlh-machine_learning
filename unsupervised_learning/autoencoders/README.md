# Autoencoders

Exercises on autoencoders with Keras.

## Tasks

| File | Description |
|------|-------------|
| `0-vanilla.py` | `autoencoder(input_dims, hidden_layers, latent_dims)` builds a vanilla autoencoder and returns the encoder, decoder and full model |
| `0-main.py` | Trains the vanilla autoencoder on MNIST for 50 epochs and plots reconstructions |
| `3-variational.py` | `autoencoder(input_dims, hidden_layers, latent_dims)` builds a variational autoencoder. The encoder returns the latent sample, the mean and the log variance |
| `3-main.py` | Trains the variational autoencoder on MNIST for 50 epochs and plots reconstructions and the latent space |

## Requirements

- Python 3.9+
- TensorFlow 2.x
- matplotlib (only for the main scripts)

## Usage

    ./0-main.py
    ./3-main.py
