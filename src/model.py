import numpy as np
import pandas as p
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
import optuna
from src.data import PreprocessingTask
import luigi
from luigi.util import inherits
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import pickle

from abc import abstractmethod


class ClassifierBase(BaseEstimator, ClassifierMixin):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def predict_proba(self):
        pass


class RegressorBase(BaseEstimator, RegressorMixin):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class Sampler(BaseEstimator, TransformerMixin):
    def __init__(self, sim_positive_max, sim_positive_min, sim_negative_max, sim_negative_min, n_sample=10000):
        sim_negative_max = sim_negative_max
        sim_negative_min = sim_negative_min
        sim_positive_max = sim_positive_max
        sim_positive_min = sim_positive_min
        n_sample = n_sample

    def fit_transform(self, X, y=None):
        if y is not None:
            positive_index = np.where(y > self.sim_positive_min & y < self.sim_positive_max)
            negative_index = np.where(y > self.sim_negative_min & y < self.sim_negative_max)

            positive_sampling = np.random.sample(positive_index, self.n_sample)
            negative_sampling = np.random.sample(negative_index, self.n_sample)

            sampling = np.concatenate([positive_sampling, negative_sampling])

            X = X[sampling, :]
            y = y[sampling]

            return X, y
        else:
            return X


####### WIP
class SimilarWeatherRegression(RegressorBase):
    def __init__(self, classifier_model):
        self.cls = classifier_model

    def fit(self, X, y):
        self.cls.fit(X, y)

    def fit_transform(self, X, y):
        p = self.cls.predict(X)
        
        return y[p]


@inherits(PreprocessingTask)
class ModelTrainingTask(luigi.Task):
    sim_negative_max = luigi.FloatParameter()
    sim_negative_min = luigi.FloatParameter()
    sim_positive_max = luigi.FloatParameter()
    sim_positive_min = luigi.FloatParameter()
    n_sample = luigi.IntParameter() 

    def requires(self):
        return self.clone_parent()

    def output(self):
        return luigi.LocalTarget("model/trained_model.pkl", foramt=luigi.format.Nop)
    
    def run(self):
        inputs = self.input()
        data = {}
        for k, target in inputs.items():
            with target.open('r') as f:
                data[k] = pickle.load(f)

        self.X = data['weather']
        self.y = data['corr']

        study = optuna.create_study()
        study.optimize(self._evaluate_model(), trial=1000)
        cls_params = study.best_params

        sampling_params = self._get_sampling_params()
        pipeline = self._get_pipeline(sampling_params, cls_params)
        pipeline.fit(self.X, self.y)
        
        target = self.output()
        with target.open('w') as f:
            pickle.dump(pipeline, f)

    def _get_sampling_params(self):
        return  dict(sim_negative_max=self.sim_negative_max,
                     sim_negative_min=self.sim_negative_min,
                     sim_positive_max=self.sim_positive_max,
                     sim_positive_min=self.sim_positive_min,
                     n_sample=self.n_sample) 

    def _get_cls_params(self, trials):
        return dict(learning_rate=trials.suggest_uniform('learning_rate', 1e-3, 1.0),
                    max_leaf_nodes=trials.suggest_int('max_leaf_nodes', 10, 50),
                    l2_reguralization=trials.suggest_uniform('l2_regularization', 0, 1),
                    max_depth=trials.suggest_int('max_depth', 1, 64))

    def _get_pipeline(self, sampler_params, cls_params):
        sampler = Sampler(**sampler_params)
        cls_model = HistGradientBoostingClassifier(**cls_params)
        model = Pipeline([('sampler', sampler), ('cls', cls_model)])

        return model

    def _evaluate_model(self):
        def objective(trials):
            sampling_params = self._get_sampling_params()
            cls_params = self._get_cls_params(trials)
            pipeline = self._get_pipeline(sampling_params, cls_params)
            score = cross_val_score(pipeline, self.X, self.y)
            return score

        return objective
