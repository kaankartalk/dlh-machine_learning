#!/usr/bin/env python3
"""Module that performs Q-learning on the FrozenLake environment"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning

    env: the FrozenLakeEnv instance
    Q: numpy.ndarray containing the Q-table
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: the learning rate
    gamma: the discount rate
    epsilon: the initial threshold for epsilon greedy
    min_epsilon: the minimum value that epsilon should decay to
    epsilon_decay: the decay rate for updating epsilon between episodes

    When the agent falls in a hole, the reward is updated to be -1

    Returns: Q, total_rewards
        Q is the updated Q-table
        total_rewards is a list containing the rewards per episode
    """
    total_rewards = []
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)

            if terminated and reward == 0:
                reward = -1

            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action])

            episode_reward += reward
            state = new_state

            if terminated or truncated:
                break

        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode)
        total_rewards.append(episode_reward)

    return Q, total_rewards
