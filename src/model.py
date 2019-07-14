import numpy as np
import pandas as p
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
import optuna
from src.data import PreprocessingTask
from src.sampler import Sampler
import luigi
from luigi.util import inherits
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
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


@inherits(PreprocessingTask)
class ModelTrainingTask(luigi.Task):
    sim_negative_max = luigi.FloatParameter()
    sim_negative_min = luigi.FloatParameter()
    sim_positive_max = luigi.FloatParameter()
    sim_positive_min = luigi.FloatParameter()
    n_samples = luigi.IntParameter() 

    def requires(self):
        return self.clone_parent()

    def output(self):
        return luigi.LocalTarget("model/trained_model.pkl", format=luigi.format.Nop)
    
    def run(self):
        inputs = self.input()
        data = {}
        for k, target in inputs.items():
            with target.open('r') as f:
                data[k] = pickle.load(f)

        self.X = data['weather']
        self.y = data['corr']
        self.sampler = Sampler(**self._get_sampling_params())

        study = optuna.create_study(direction='maximize')
        study.optimize(self._evaluate_model(), n_trials=1)
        cls_params = study.best_params

        classifier= self._get_classifier(cls_params)
        label = self.sampler.correlation_to_binary(self.y)
        classifier.fit(self.X, label)
        
        target = self.output()
        with target.open('w') as f:
            pickle.dump(classifier, f)

    def _get_sampling_params(self):
        return  dict(sim_negative_max=self.sim_negative_max,
                     sim_negative_min=self.sim_negative_min,
                     sim_positive_max=self.sim_positive_max,
                     sim_positive_min=self.sim_positive_min,
                     n_samples=self.n_samples) 

    def _get_cls_params(self, trials):
        return dict(learning_rate=trials.suggest_uniform('learning_rate', 1e-3, 1.0),
                    max_leaf_nodes=trials.suggest_int('max_leaf_nodes', 10, 50),
                    l2_regularization=trials.suggest_uniform('l2_regularization', 0, 1),
                    max_depth=trials.suggest_int('max_depth', 1, 64))

    def _get_classifier(self, cls_params):
        cls_model = HistGradientBoostingClassifier(**cls_params)

        return cls_model 

    def _evaluate_model(self):
        def objective(trials):
            X_sampled, y_sampled = self.sampler.sample(self.X, self.y)
            cls_params = self._get_cls_params(trials)
            classifier = self._get_classifier(cls_params)
            li_score = []
            for train_index, test_index in KFold(n_splits=5).split(X_sampled):
                classifier.fit(X_sampled[train_index], y_sampled[train_index])
                li_score.append(classifier.score(X_sampled[test_index], y_sampled[test_index]))

            return np.mean(li_score) 

        return objective
