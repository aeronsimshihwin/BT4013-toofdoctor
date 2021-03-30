import utils
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgboost
from xgboost import XGBClassifier

SAVED_DIR = "/saved_models/categorical/xgb"
FUTURES_LIST = utils.futuresList

class XGBWrapper:
    SAVED_DIR = 'saved_models/categorical/xgb/val/pct'
    # SAVED_DIR = 'saved_models/categorical/xgb/val/pct_tech'
    # SAVED_DIR = 'saved_models/categorical/xgb/val/pct_macro'
    # SAVED_DIR = 'saved_models/categorical/xgb/val/pct_tech_macro'
    
    def __init__(self, model=None, y=None, X=None):
        self.model = model
        self.get_y = y # Extracts y from data[future]
        self.get_X = X # Extracts X from data[future]

    def fit(self, data, future, **kwargs):
        if future in data:
            self._fit(*self._y_X(data, future), **kwargs)

    def predict(self, data, future, **kwargs):
        try:
            y, X = self._y_X(data, future)
            y_pred = self.model.predict_proba(X)
            y_pred_pos = y_pred[:, 1][-1] # get probability of class 1
            y_pred_norm = y_pred_pos - 0.5 # normalise to include long and short
            y_pred_norm_long = max(0,y_pred_norm) # long only
            return y_pred_norm_long # returns only last value
        except:
            return 0 # input invalid

    def _y_X(self, data, future):
        data_df = data[future]
        
        # slice only relevant data
        X = data_df[self.get_X]
        y = data_df[self.get_y]

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
