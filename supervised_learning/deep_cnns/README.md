# Deep CNNs

This directory contains implementations of building blocks and architectures from
"Deep Residual Learning for Image Recognition" (2015), written with TensorFlow Keras.

## Requirements

- Python 3
- TensorFlow (Keras)
- All files follow the `#!/usr/bin/env python3` shebang and are executable

## Files

| File | Description |
|------|-------------|
| `2-identity_block.py` | `identity_block(A_prev, filters)` builds an identity block |
| `3-projection_block.py` | `projection_block(A_prev, filters, s=2)` builds a projection block |
| `4-resnet50.py` | `resnet50()` builds the ResNet-50 architecture |
| `2-main.py` | Test file for the identity block |
| `3-main.py` | Test file for the projection block |
| `4-main.py` | Test file for ResNet-50 |

## Details

- All convolutions are followed by batch normalization along the channels axis, then ReLU
- All weights use He normal initialization with `seed=0`
- `resnet50()` expects input of shape `(224, 224, 3)` and uses `identity_block` and `projection_block`

## Usage

```bash
./2-main.py
./3-main.py
./4-main.py
```

Expected total parameters:

- Identity block: 71,552
- Projection block: 57,408
- ResNet-50: 25,636,712
