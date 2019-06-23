import luigi
import argparse

from src.utils import load_data
from src.data import PreprocessingTask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='search method for similar weather days.')
    parser.add_argument('--weather_csv', type=str, required=True)
    parser.add_argument('--demand_csv', type=str, required=True)
    args = parser.parse_args()

    luigi.build([PreprocessingTask(demand_filepath=args.demand_csv, weather_filepath=args.weather_csv)], local_scheduler=True)