# Optimization

Optimization techniques for training neural networks, implemented from
scratch in NumPy and with their TensorFlow equivalents.

## Learning Objectives

- What is a hyperparameter and how to choose one
- Feature scaling and standardization
- Mini-batch gradient descent
- Moving averages and bias correction
- Gradient descent with momentum, RMSProp and Adam
- Learning rate decay
- Batch normalization

## Requirements

- Ubuntu 20.04 LTS / Python 3.9
- numpy 1.25.2
- tensorflow 2.15
- pycodestyle 2.11.1
- All files are executable and end with a new line
- All modules, classes and functions are documented

## Tasks

| File | Function | Description |
|------|----------|-------------|
| `0-norm_constants.py` | `normalization_constants(X)` | Mean and standard deviation of each feature |
| `1-normalize.py` | `normalize(X, m, s)` | Standardizes a matrix |
| `2-shuffle_data.py` | `shuffle_data(X, Y)` | Shuffles two matrices the same way |
| `3-mini_batch.py` | `create_mini_batches(X, Y, batch_size)` | Builds mini-batches for training |
| `4-moving_average.py` | `moving_average(data, beta)` | Weighted moving average with bias correction |
| `5-momentum.py` | `update_variables_momentum(...)` | Momentum update in NumPy |
| `6-momentum.py` | `create_momentum_op(alpha, beta1)` | Momentum optimizer in TensorFlow |
| `7-RMSProp.py` | `update_variables_RMSProp(...)` | RMSProp update in NumPy |
| `8-RMSProp.py` | `create_RMSProp_op(alpha, beta2, epsilon)` | RMSProp optimizer in TensorFlow |
| `9-Adam.py` | `update_variables_Adam(...)` | Adam update in NumPy |
| `10-Adam.py` | `create_Adam_op(alpha, beta1, beta2, epsilon)` | Adam optimizer in TensorFlow |
| `11-learning_rate_decay.py` | `learning_rate_decay(...)` | Inverse time decay in NumPy |
| `12-learning_rate_decay.py` | `learning_rate_decay(...)` | Inverse time decay in TensorFlow |
| `13-batch_norm.py` | `batch_norm(Z, gamma, beta, epsilon)` | Batch normalization in NumPy |
| `14-batch_norm.py` | `create_batch_norm_layer(prev, n, activation)` | Batch normalization layer in TensorFlow |

## Usage

sad
s

x



cat > README.md <<'EOF'
# Optimization

Optimization techniques for training neural networks, in NumPy and TensorFlow.

## Requirements

- Python 3.9, numpy 1.25.2, tensorflow 2.15
- pycodestyle 2.11.1
- All files are executable and documented

## Files

| File | Description |
|------|-------------|
| `0-norm_constants.py` | Mean and standard deviation of each feature |
| `1-normalize.py` | Standardizes a matrix |
| `2-shuffle_data.py` | Shuffles two matrices the same way |
| `3-mini_batch.py` | Creates mini-batches for training |
| `4-moving_average.py` | Weighted moving average with bias correction |
| `5-momentum.py` | Momentum update (NumPy) |
| `6-momentum.py` | Momentum optimizer (TensorFlow) |
| `7-RMSProp.py` | RMSProp update (NumPy) |
| `8-RMSProp.py` | RMSProp optimizer (TensorFlow) |
| `9-Adam.py` | Adam update (NumPy) |
| `10-Adam.py` | Adam optimizer (TensorFlow) |
| `11-learning_rate_decay.py` | Inverse time decay (NumPy) |
| `12-learning_rate_decay.py` | Inverse time decay (TensorFlow) |
| `13-batch_norm.py` | Batch normalization (NumPy) |
| `14-batch_norm.py` | Batch normalization layer (TensorFlow) |

## Author

Kaan Kartalkuyucu
