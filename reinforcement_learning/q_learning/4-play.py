#!/usr/bin/env python3
"""Module that has the trained agent play an episode"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode

    env: the FrozenLakeEnv instance
    Q: numpy.ndarray containing the Q-table
    max_steps: maximum number of steps in the episode

    The agent always exploits the Q-table

    Returns: the total rewards for the episode and a list of rendered
        outputs representing the board state at each step
    """
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = [env.render()]

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward
        rendered_outputs.append(env.render())

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
