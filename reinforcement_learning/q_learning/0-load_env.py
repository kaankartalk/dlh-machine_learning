#!/usr/bin/env python3
"""Module that loads the FrozenLake environment"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv from gymnasium

    desc: None or list of lists containing a custom map description
    map_name: None or string containing the pre-made map to load
    is_slippery: boolean, determines if the ice is slippery

    If both desc and map_name are None, a random 8x8 map is loaded

    Returns: the environment
    """
    return gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
