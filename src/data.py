import luigi
from luigi.util import inherits

import pickle
from utils import load_data
from preprocessing import demand_correlation


class LocalDataLoadTask(luigi.Task):
    demand_filepath = luigi.Parameter()
    weather_filepath = luigi.Parameter()

    def input(self):
        df_demand, df_weather = load_data(self.cons_csv_path, self.weather_csv_path)

        return dict(demand=df_demand,
                    weather=df_weather)

    def output(self):
        return dict(demand=luigi.LocalTarget("data/demand.pkl"),
                    weather=luigi.LocalTarget("data/weahter.pkl"))

    def run(self):
        targets = self.output()

        inputs = self.input()
        for k, target in targets.items():
            with target.open('w') as f:
                pickle.dump(inputs[k], f)


@inherits(LocalDataLoadTask)
class PreprocessingTask(luigi.Task):
    def requires(self):
        return self.clone_parent()

    def output(self):
        return luigi.LocalTarget("data/corr.pkl")

    def run(self):
        
        with self.input().open('r') as f:
            inputs = pickle.load(f)

        df_demand = inputs['demand']
        corr = demand_correlation(df_demand)

        with self.output().open('w') as f:
            pickle.dump(corr, f)