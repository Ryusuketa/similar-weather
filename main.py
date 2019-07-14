import luigi
import argparse
import numpy as np

from src.utils import load_data
from src.model import ModelTrainingTask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='search method for similar weather days.')
    parser.add_argument('--weather_csv', type=str, required=True)
    parser.add_argument('--demand_csv', type=str, required=True)
    parser.add_argument('--period', type=int, default=1)
    args = parser.parse_args()

    luigi.configuration.LuigiConfigParser.add_config_path('./conf/model_conf.ini')
    np.random.seed(2000)

    luigi.build([ModelTrainingTask(demand_filepath=args.demand_csv,
                                   weather_filepath=args.weather_csv,
                                   period=args.period,)],
                local_scheduler=True)