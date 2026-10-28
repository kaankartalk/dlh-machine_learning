#!/usr/bin/env python3
"""Module that implements a full training with Monte-Carlo policy gradient"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implements a full training

    env: initial environment
    nb_episodes: number of episodes used for training
    alpha: the learning rate
    gamma: the discount factor

    Returns: all values of the score (sum of all rewards during one episode)
    """
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        grads = []
        rewards = []
        done = False

        while not done:
            action, grad = policy_gradient(state, weight)
            state, reward, terminated, truncated, _ = env.step(action)
            grads.append(grad)
            rewards.append(reward)
            done = terminated or truncated

        G = 0
        for grad, reward in zip(reversed(grads), reversed(rewards)):
            G = reward + gamma * G
            weight += alpha * grad * G

        score = sum(rewards)
        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))

    return scores
