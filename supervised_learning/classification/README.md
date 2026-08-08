# Classification Using Neural Networks

Building a binary image classifier from scratch with NumPy — no deep learning
frameworks. Starts from a single neuron and grows into a network with a hidden
layer, implementing forward propagation, cost, evaluation, and gradient descent
by hand.

## Dataset

Handwritten digit images (28x28 grayscale, flattened to 784 features) labeled
`0` / `1` for binary classification.

- `../data/Binary_Train.npz` — training set
- `../data/Binary_Dev.npz` — development set

Each `.npz` holds `X` with shape `(m, 28, 28)` and `Y` with shape `(1, m)`.
Images are reshaped and transposed to `(784, m)` before training.

The data files are not tracked in this repository.

## Requirements

- Ubuntu 20.04 LTS
- Python 3.9
- NumPy 1.25.2
- `pycodestyle` 2.11.1
- All files are executable, start with `#!/usr/bin/env python3`, and end with a
  new line
- Modules, classes, and functions are documented
- Only `numpy` is imported, except where plotting requires `matplotlib`

## Files

### Neuron — a single neuron

| File | Adds |
| --- | --- |
| `0-neuron.py` | Constructor with input validation; public `W`, `b`, `A` |
| `1-neuron.py` | Private attributes with getter-only properties |
| `2-neuron.py` | `forward_prop` — sigmoid activation |
| `3-neuron.py` | `cost` — logistic regression loss |
| `4-neuron.py` | `evaluate` — predictions plus cost |
| `5-neuron.py` | `gradient_descent` — one training pass |
| `6-neuron.py` | `train` — loops gradient descent over `iterations` |
| `7-neuron.py` | `train` with `verbose`, `graph`, and `step` |

### NeuralNetwork — one hidden layer

| File | Adds |
| --- | --- |
| `8-neural_network.py` | Constructor with `nx` and `nodes`; public attributes |
| `9-neural_network.py` | Private attributes with getter-only properties |
| `10-neural_network.py` | `forward_prop` — hidden layer then output neuron |
| `11-neural_network.py` | `cost` |
| `12-neural_network.py` | `evaluate` |
| `13-neural_network.py` | `gradient_descent` — backpropagation |
| `14-neural_network.py` | `train` |
| `15-neural_network.py` | `train` with `verbose`, `graph`, and `step` |

## Usage

```bash
chmod +x *.py
./7-main.py
./15-main.py
```

## Concepts

- **Weight initialization** — weights are drawn from a random normal
  distribution to break symmetry; biases start at 0.
- **Forward propagation** — `A = sigmoid(W · X + b)` maps inputs to a
  probability in `(0, 1)`. In the two-layer network the hidden activation `A1`
  becomes the input to the output neuron.
- **Cost** — logistic regression loss, using `1.0000001 - A` in place of
  `1 - A` to avoid taking the log of zero.
- **Backpropagation** — `dZ1` depends on the current `W2`, so every gradient is
  computed before any weight is updated.
- **Gradient descent** — weights move against the gradient of the cost, scaled
  by the learning rate `alpha`.

## Results

| Model | Train accuracy | Dev accuracy |
| --- | --- | --- |
| `Neuron`, 3000 iterations | 99.67% | 99.81% |
| `NeuralNetwork` (3 nodes), 5000 iterations | 99.40% | 99.57% |

## Author

KK
