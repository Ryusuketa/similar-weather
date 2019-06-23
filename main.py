import luigi
import argparse

from src.utils import load_data
from src.preprocessing import demand_correlation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='search method for similar weather days.')

    df_demand, df_weather = load_data('./localfile/demand.csv', './localfile/data_gathered.csv')
    demand_correlation(df_demand)