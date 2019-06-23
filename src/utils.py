import luigi
import numpy as np
import pandas as pd


def check_cons_columns(df: pd.DataFrame):
    if set(df.columns) != {'datetime', 'demand'}:
        raise ValueError('The column names must be "datetime" and "consumption". Please modify column names.')
    else:
        return True


def check_weather_columns(df: pd.DataFrame):
    if set(df.columns) != {'datetime', 'consumption'}:
        raise ValueError('The column names must be "datetime" and "consumption". Please modify column names.')
    else:
        return True


def load_data(cons_csv_path: str, weather_csv_path: str) -> pd.DataFrame:
    df_demand = pd.read_csv(cons_csv_path)
    check_cons_columns(df_demand) 
    df_weather = pd.read_csv(weather_csv_path)

    return df_demand, df_weather


def get_datetime2index(li_datetime: list) -> pd.DataFrame:
    return {dt: i for i, dt in zip(range(li_datetime), dt)}


def get_train_validation_data(corr, features, window=1, val_rate=0.2):
    val_num = int(corr.shape[0] * 0.2)

    train_corr = corr[:-val_num:,:-val_num]
    val_corr = corr[-val_num:,-val_num:]

    train_features = features[:-val_num, :-val_num]
    val_features = features[-val_num:, -val_num:]

    return train_corr, val_corr, train_features, val_features