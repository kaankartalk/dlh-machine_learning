#!/usr/bin/env python3
"""Selects the last 10 rows of High and Close as a numpy.ndarray"""


def array(df):
    """
    Selects the last 10 rows of the High and Close columns and
    converts them into a numpy.ndarray

    df: pd.DataFrame containing columns named High and Close
    Returns: the numpy.ndarray
    """
    A = df[["High", "Close"]].tail(10).to_numpy()
    return A
