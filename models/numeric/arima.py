from datetime import datetime

import numpy as np
import pmdarima as pm

class ArimaWrapper:
    SAVED_DIR = None # Override in subclass
    
    def __init__(self, model=None, y=None, X=None):
        self.model = model
        self.get_y = y # Extracts y from data[future]
        self.get_X = X # Extracts X from data[future]
    
    def fit(self, data, future, **kwargs):
        if future in data:
            self._fit(*self._y_X(data, future), **kwargs)
    
    def predict(self, data, future, **kwargs):
        try:
            return self._predict(*self._y_X(data, future), **kwargs)
        except:
            return np.nan
    
    def _y_X(self, data, future):
        X = None if self.get_X is None else self.get_X(data[future])
        y = None if self.get_y is None else self.get_y(data[future])
        return y, X
    
    def _fit(self, y, X, **kwargs):
        if self.model is None:
            self.model = pm.auto_arima(y, **kwargs)
        else:
            self.model.update(y.iloc[-1])
    
    def _predict(self, y, X, **kwargs):
        self._fit(y, X, **kwargs)
        y_pred = self.model.predict(n_periods=1)[0]
        y_diff = y_pred - y.iloc[-1]
        return y_diff

class ArimaRaw(ArimaWrapper):
    SAVED_DIR = 'saved_models/numeric/arima'
    
    def __init__(self):
        super().__init__(y=self.transform_predictor)
    
    def transform_predictor(self, data):
        """Returns the last 50 close prices in a train split"""
        rows = data.index < datetime(2021, 1, 1)
        cols = 'CLOSE'
        y = data.loc[rows, cols]
        return y

class ArimaLinear(ArimaWrapper):
    SAVED_DIR = 'saved_models/numeric/arimalinear'
    
    def __init__(self):
        super().__init__(y=self.transform_predictor)
    
    def transform_predictor(self, data):
        rows = data.index < datetime(2021, 1, 1)
        cols = 'CLOSE'
        y = data.loc[rows, cols]
        y = np.log(y)
        return y

class ArimaNoTrend(ArimaWrapper):
    SAVED_DIR = 'saved_models/numeric/arimanotrend'
    
    def __init__(self):
        super().__init__(y=self.transform_predictor)
    
    def transform_predictor(self, data):
        rows = data.index < datetime(2021, 1, 1)
        cols = 'CLOSE'
        y = data.loc[rows, cols]
        y = np.diff(y).dropna()
        return y

class ArimaLinearNoTrend(ArimaWrapper):
    SAVED_DIR = 'saved_models/numeric/arimalinearnotrend'
    
    def __init__(self):
        super().__init__(y=self.transform_predictor)
    
    def transform_predictor(self, data):
        rows = data.index < datetime(2021, 1, 1)
        cols = 'CLOSE'
        y = data.loc[rows, cols]
        y = np.log(y)
        y = y.diff().dropna()
        return y
