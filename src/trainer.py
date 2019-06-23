import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import List

from sklearn.model_selection import KFold


class ClassifierValidation(object):
    def __init__(self, model, n_split=5):
        self.kf = KFold(n_split)
        self.model = model

    def set_data(self, X, y):
        self.X = X
        self.y = y

    def _train(self, X_train, y_train, X_val):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict_proba(X_val)

        return y_pred
    
    def validation_run(self):
        result = []
        for train, val in self.kf.split(self.X):
            X_train, y_train = self.X[train], self.y[train]
            X_val, y_val = self.X[val], self.y[val]

            y_pred = self._train(X_train, y_train, X_val)
            result.append(pd.DataFrame(y_pred, index=val))

        self.y_pred = pd.concat(result, axis=0).sort_index()

    def report(self, report_path):
        df = pd.DataFrame({'output_predict': self.y_pred, 'output_true': self.y})
        
        df.to_csv(report_path)
