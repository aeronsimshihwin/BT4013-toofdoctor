from datetime import datetime

import utils
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class LogRegWrapper:
    SAVED_DIR = 'saved_models/categorical/logreg/perc'
    
    def __init__(self, model=None, y=None, X=None):
        self.model = model
        self.get_y = y # Extracts y from data[future]
        self.get_X = X # Extracts X from data[future]

    def fit(self, data, future, **kwargs):
        if future in data:
            self._fit(*self._y_X(data, future), **kwargs)

    def predict(self, data, future, **kwargs):
        return self._predict(*self._y_X(data, future), **kwargs)
        # try:
        #     return self._predict(*self._y_X(data, future), **kwargs)
        # except:
        #     return np.nan

    def _y_X(self, data, future):
        data_df = data[future]
        # remove na and zero values
        data_df = data_df[(data_df.VOL != 0) & (data_df.CLOSE != 0)]
        data_df = data_df.dropna(axis=0)
        # generate perc close and perc vol (features)
        X = utils.generate_X_df([data_df.CLOSE, data_df.VOL], ["perc", "perc"])
        # generate target
        y = utils.generate_y_cat(data_df.CLOSE)
        # drop na
        X = X.dropna(axis=0)
        y = y.dropna(axis=0)
        # get intersection of all dataframes
        common_index = X.index.intersection(y.index)
        X = X[X.index.isin(common_index)].to_numpy()
        y = y[y.index.isin(common_index)].to_numpy(np.dtype(int))
        return y, X

    def _fit(self, y, X, **kwargs):
        if self.model is None:
            model = LogisticRegression(**kwargs)
            fitted = model.fit(X, y)
            self.model = fitted
    
    def _predict(self, y, X, **kwargs):
        X_test = X[-1]
        y_pred = self.model.predict_proba(X_test.reshape(1, -1)) # returns [P(-1), P(1)]
        y_pred_pos = y_pred[0][1] # return probability of class 1
        y_pred_norm = y_pred_pos - 0.5 # normalise, centre about zero
        y_pred_norm_pos = max(0, y_pred_norm)
        return y_pred_pos ## to review this
