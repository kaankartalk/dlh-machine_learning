#!/usr/bin/env python3
"""Module that defines a function to prune a pd.DataFrame."""


def prune(df):
    """Removes entries where Close has NaN values.

    Args:
        df: pd.DataFrame containing a column named Close.

    Returns:
        The modified pd.DataFrame.
    """
    df = df.dropna(subset=['Close'])

    return df
