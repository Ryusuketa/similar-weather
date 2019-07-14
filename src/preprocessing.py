import numpy as np
import pandas as pd
from itertools import combinations


def demand_correlation(df, period=1):
    if period < 1:
        raise ValueError("Input period have to be over 1")

    df = df.copy()
    demand = df['demand'].values

    df['demand'] = (demand - demand.mean()) / demand.std()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    dates = pd.date_range(df.index[0].strftime('%Y-%m-%d'),  df.index[-1].strftime('%Y-%m-%d'))
    dates = [d.strftime('%Y-%m-%d') for d in dates]
    dates = dates[period-1:]
    data = np.array([df[date]['demand'].values for date in dates])

    corr = np.corrcoef(data)
    return corr


def get_weather_vector(df_weather, period=1, categorical_columns=['weather']):
    if period < 1:
        raise ValueError("Input period have to be over 1")
    p_gap = period-1
    df = df_weather.copy()
    df = pd.get_dummies(df, columns=categorical_columns,
                        prefix=categorical_columns)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    dates = pd.date_range(df.index[0].strftime('%Y-%m-%d'),
                          df.index[-1].strftime('%Y-%m-%d'))
    dates = [d.strftime('%Y-%m-%d') for d in dates]

    data = np.array([df[dates[idx]:dates[idx+p_gap]].values.flatten()
                     for idx in range(len(dates)-p_gap)]).T
    data = data[:, :, np.newaxis]
    data = data.repeat(len(dates)-p_gap, 2)
    t_data = data.transpose(0, 2, 1)
    return np.concatenate([data, t_data], 0)


def get_data_pairs(corr: np.ndarray, weather: np.ndarray):
    def _data_pairs(arr_len: int):
        return combinations(np.arange(arr_len), 2)
    
    arr_len = len(corr)
    corr = np.array([corr[idx] for idx in _data_pairs(arr_len)])
    weather = np.array([weather[:, idx[0], idx[1]]
                        for idx in _data_pairs(arr_len)])

    return corr, weather