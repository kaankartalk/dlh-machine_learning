# Autoencoders

Exercises on autoencoders with Keras.

## Tasks

| File | Description |
|------|-------------|
| `0-vanilla.py` | `autoencoder(input_dims, hidden_layers, latent_dims)` builds a vanilla autoencoder and returns the encoder, decoder and full model |
| `0-main.py` | Trains the vanilla autoencoder on MNIST for 50 epochs and plots reconstructions |
| `1-sparse.py` | `autoencoder(input_dims, hidden_layers, latent_dims, lambtha)` builds a sparse autoencoder with L1 regularization on the encoded output |
| `1-main.py` | Trains the sparse autoencoder on MNIST for 50 epochs and plots reconstructions |
| `2-convolutional.py` | `autoencoder(input_dims, filters, latent_dims)` builds a convolutional autoencoder with max pooling in the encoder and upsampling in the decoder |
| `2-main.py` | Trains the convolutional autoencoder on MNIST for 50 epochs and plots reconstructions |
| `3-variational.py` | `autoencoder(input_dims, hidden_layers, latent_dims)` builds a variational autoencoder. The encoder returns the latent sample, the mean and the log variance |
| `3-main.py` | Trains the variational autoencoder on MNIST for 50 epochs and plots reconstructions and the latent space |

## Requirements

- Python 3.9+
- TensorFlow 2.x
- matplotlib (only for the main scripts)

## Usage

    ./0-main.py
    ./1-main.py
    ./2-main.py
    ./3-main.py
