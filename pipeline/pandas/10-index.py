#!/usr/bin/env python3
"""Module that defines a function to set the index of a pd.DataFrame."""


def index(df):
    """Sets the Timestamp column as the index of the dataframe.

    Args:
        df: pd.DataFrame containing a Timestamp column.

    Returns:
        The modified pd.DataFrame.
    """
    df = df.set_index('Timestamp')

    return df
