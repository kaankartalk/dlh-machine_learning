#!/usr/bin/env python3
"""Sorts a pd.DataFrame by the High price in descending order"""


def high(df):
    """
    Sorts the pd.DataFrame by the High price in descending order

    df: pd.DataFrame to sort
    Returns: the sorted pd.DataFrame
    """
    df = df.sort_values(by="High", ascending=False)
    return df
