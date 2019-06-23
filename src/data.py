import luigi
from luigi.util import inherits

import pickle
from src.utils import load_data
from src.preprocessing import demand_correlation, get_weather_vector
import numpy as np

from itertools import combinations

class LocalDataLoadTask(luigi.Task):
    demand_filepath = luigi.Parameter()
    weather_filepath = luigi.Parameter()

    def input(self):
        df_demand, df_weather = load_data(self.demand_filepath, self.weather_filepath)

        return dict(demand=df_demand,
                    weather=df_weather)

    def output(self):
        return dict(demand=luigi.LocalTarget("data/demand.pkl", format=luigi.format.Nop),
                    weather=luigi.LocalTarget("data/weather.pkl", format=luigi.format.Nop))

    def run(self):
        targets = self.output()

        inputs = self.input()
        for k, target in targets.items():
            with target.open('wb') as f:
                pickle.dump(inputs[k],f)


@inherits(LocalDataLoadTask)
class PreprocessingTask(luigi.Task):
    def requires(self):
        return self.clone_parent()

    def output(self):
        return dict(corr=luigi.LocalTarget("data/corr.pkl", format=luigi.format.Nop),
                    weather=luigi.LocalTarget("data/weather_feature.pkl", format=luigi.format.Nop))

    def run(self):
        
        inputs = self.input()
        data = {}
        for k, target in inputs.items():
            with target.open('r') as f:
                data[k] = pickle.load(f)

        df_demand = data['demand']
        df_weather = data['weather']
        corr = demand_correlation(df_demand)
        weather = get_weather_vector(df_weather)

        print(weather.shape)
        data_pairs = combinations(np.arange(len(corr)), 2)
        
        corr = np.array([corr[idx] for idx in data_pairs])
        data = dict(corr=corr, weather=weather)
        for k, target in self.output().items():
            with target.open('w') as f:
              pickle.dump(data[k], f)