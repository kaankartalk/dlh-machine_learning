#!/usr/bin/env python3
"""Renames the Timestamp column to Datetime and converts values"""
import pandas as pd


def rename(df):
    """
    Renames the Timestamp column to Datetime, converts values to
    datetime, and displays only the Datetime and Close columns
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    df = df[["Datetime", "Close"]]
    return df
