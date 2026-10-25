# Attention

This directory contains tasks on the attention mechanism and transformers,
written in Python 3 with numpy and tensorflow.

## Requirements

* Python 3.9
* numpy 1.25.2
* tensorflow 2.15
* All files follow pycodestyle 2.11.0
* All files are executable and end with a new line
* All modules, classes and functions have documentation

## Files

| File | Description |
| --- | --- |
| `4-positional_encoding.py` | Function `positional_encoding(max_seq_len, dm)` that calculates the positional encoding for a transformer |
| `5-sdp_attention.py` | Function `sdp_attention(Q, K, V, mask=None)` that calculates the scaled dot product attention |
| `6-multihead_attention.py` | Class `MultiHeadAttention` that performs multi head attention |
| `7-transformer_encoder_block.py` | Class `EncoderBlock` that creates an encoder block for a transformer |

## Author

Kaan Kartal Kuyucu
