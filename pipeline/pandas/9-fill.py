#!/usr/bin/env python3
"""Module that defines a function to fill missing values in a pd.DataFrame."""


def fill(df):
    """Fills missing values in a pd.DataFrame.

    Args:
        df: pd.DataFrame containing columns Open, High, Low, Close,
            Volume_(BTC), Volume_(Currency), and Weighted_Price.

    Returns:
        The modified pd.DataFrame.
    """
    df = df.drop(columns=['Weighted_Price'])

    df['Close'] = df['Close'].ffill()

    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
