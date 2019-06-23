import numpy as np
import pandas as pd


def demand_correlation(df):
    df = df.copy()
    demand = df['demand'].values

    df['demand'] = (demand - demand.mean()) / demand.std()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    dates = pd.date_range(df.index[0].strftime('%Y-%m-%d'),  df.index[-1].strftime('%Y-%m-%d'))
    dates = [d.strftime('%Y-%m-%d') for d in dates]
    data = np.array([df[date]['demand'].values for date in dates])

    corr = np.corrcoef(data)

    return corr

def get_correlation_label(corr, threshold=0.7):
    pass
