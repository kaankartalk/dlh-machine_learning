# Q-learning

Reinforcement learning project using gymnasium (FrozenLake-v1) and numpy.

## Requirements

- Python 3.9
- numpy 1.25.2
- gymnasium 0.29.1
- pycodestyle 2.11.1

## Files

| File | Description |
|------|-------------|
| `0-load_env.py` | `load_frozen_lake(desc=None, map_name=None, is_slippery=False)` loads the pre-made FrozenLake environment |
| `1-q_init.py` | `q_init(env)` initializes the Q-table as a numpy.ndarray of zeros |
| `2-epsilon_greedy.py` | `epsilon_greedy(Q, state, epsilon)` uses epsilon-greedy to determine the next action |
| `3-q_learning.py` | `train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)` performs Q-learning and returns the Q-table and rewards per episode |
