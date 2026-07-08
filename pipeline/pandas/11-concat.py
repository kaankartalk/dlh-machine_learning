#!/usr/bin/env python3
"""Module that defines a function to concatenate two pd.DataFrame objects."""
import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """Concatenates two pd.DataFrame objects indexed on Timestamp.

    Args:
        df1: pd.DataFrame representing coinbase data.
        df2: pd.DataFrame representing bitstamp data.

    Returns:
        The concatenated pd.DataFrame, with df2 rows up to and
        including timestamp 1417411920 placed above df1 rows,
        labeled with keys 'bitstamp' and 'coinbase' respectively.
    """
    df1 = index(df1)
    df2 = index(df2)

    df2 = df2[df2.index <= 1417411920]

    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    return df
