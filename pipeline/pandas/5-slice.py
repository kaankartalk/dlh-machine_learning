#!/usr/bin/env python3
"""Slices a pd.DataFrame, selecting every 60th row"""


def slice(df):
    """
    Extracts specific columns and selects every 60th row

    df: pd.DataFrame to slice
    Returns: the sliced pd.DataFrame
    """
    df = df[["High", "Low", "Close", "Volume_(BTC)"]]
    df = df[::60]
    return df