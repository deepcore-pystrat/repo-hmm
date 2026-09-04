import pandas as pd

from data_loader.loader_data import DataConfig
import numpy as np


def load_spot_sugar(config: DataConfig) -> pd.DataFrame:
    data = pd.read_excel(config.data_path, header=2, index_col='DATE')
    return data, config.asset_name

def format_spot_sugar(data: pd.DataFrame, asset_name: str) -> pd.DataFrame:

    data.index = pd.to_datetime(data.index)
    data[f'MID_{asset_name}'] = (data['BID'] + data['OFFER']) / 2
    data.drop(columns=['FUTURES', 'BID', 'OFFER'], inplace=True)
    data.ffill(inplace=True)

    return data

def get_sugar_data(config: DataConfig) -> pd.DataFrame:
    spot_data, asset_name = load_spot_sugar(config)
    formatted_spot_data = format_spot_sugar(spot_data, asset_name)
    return formatted_spot_data



def load_futures_sugar(config: DataConfig) -> pd.DataFrame:
    data = pd.read_excel(config.data_path, index_col='Date')
    return data



def get_bid_offer_data(cfg: DataConfig) -> pd.DataFrame:
    """
    Load custom bid/offer Excel data and build MID / BA_SPREAD.

    Expected columns in Excel:
    - DATE
    - FUTURES
    - BID
    - OFFER
    """
    df = pd.read_excel(cfg.data_path, skiprows=1)
    df.columns = df.columns.str.strip()

    required_cols = ["DATE", "FUTURES", "BID", "OFFER"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["BID"] = pd.to_numeric(df["BID"], errors="coerce")
    df["OFFER"] = pd.to_numeric(df["OFFER"], errors="coerce")

    df = df.dropna(subset=["DATE", "BID", "OFFER"])
    df = df.sort_values("DATE").set_index("DATE")

    # remove duplicated dates if any
    df = df[~df.index.duplicated(keep="first")]

    # keep only valid quotes
    df = df[(df["BID"] > 0) & (df["OFFER"] > 0)]

    df["MID"] = (df["BID"] + df["OFFER"]) / 2.0
    df["BA_SPREAD"] = df["OFFER"] - df["BID"]

    # safety
    df = df[(df["MID"] > 0) & (df["BA_SPREAD"] >= 0)]

    return df[["FUTURES", "BID", "OFFER", "MID", "BA_SPREAD"]]



