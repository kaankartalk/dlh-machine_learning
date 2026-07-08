#!/usr/bin/env python3
"""Module that defines a function to rearrange a MultiIndex pd.DataFrame."""
import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """Concatenates two pd.DataFrame objects with Timestamp as the
    first level of the MultiIndex.

    Args:
        df1: pd.DataFrame representing coinbase data.
        df2: pd.DataFrame representing bitstamp data.

    Returns:
        The concatenated pd.DataFrame, indexed by Timestamp first
        and then by source, in chronological order.
    """
    df1 = index(df1)
    df2 = index(df2)

    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    df = df.reorder_levels([1, 0])

    df = df.sort_index()

    return df
