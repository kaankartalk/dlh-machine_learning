#!/usr/bin/env python3
"""Sorts a pd.DataFrame in reverse chronological order and transposes it"""


def flip_switch(df):
    """
    Sorts the data in reverse chronological order and transposes it

    df: pd.DataFrame to transform
    Returns: the transformed pd.DataFrame
    """
    df = df.sort_index(ascending=False)
    df = df.transpose()
    return df
