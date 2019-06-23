import numpy as np
import pandas as p
from sklearn.base import BaseEstimator, ClassifierMixin

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


def get_model(model_name):
    pass
