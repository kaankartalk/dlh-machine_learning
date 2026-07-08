#!/usr/bin/env python3
"""Module that defines a function to compute descriptive statistics
of a pd.DataFrame.
"""


def analyze(df):
    """Computes descriptive statistics for all columns except Timestamp.

    Args:
        df: pd.DataFrame that may contain a Timestamp column.

    Returns:
        A new pd.DataFrame containing the descriptive statistics.
    """
    df = df.drop(columns=['Timestamp'], errors='ignore')

    stats = df.describe()

    return stats
