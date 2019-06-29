import numpy as np
import pandas as p
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.emsenble import HistGradientBoostingClassifier

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

    @abstractmethod
    def predict_proba(self):
        pass


class SimilarWeatherRegression(RegressorBase):
    def __init__(self, classifier_model, past_data):
        pass


class PredictResidual(RegressorBase):
    def __init__(self, pattern_regression):
        pass


def get_pipeline():
    cls_pipeline = [('HGB': HistGradientBoostingClassifier())]
    rgr_pipeline = [('SWR': SimilarWeatherRegression(cls_pipeline))]
