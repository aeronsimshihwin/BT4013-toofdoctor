from datetime import datetime

import numpy as np
import pmdarima as pm

class ArimaWrapper:
    SAVED_DIR = None # Override in subclass
    
    def __init__(self, model=None, y=None, X=None, y_pred=None):
        self.model = model
        self.y = y # Extracts y from data[future]
        self.X = X # Extracts X from data[future]
        self.y_pred = y_pred # Transform output
    
    def fit(self, data, future, **kwargs):
        if future in data:
            self._fit(*self._get_vars(data, future), **kwargs)
    
    def predict(self, data, future, **kwargs):
        try:
            return self._predict(*self._get_vars(data, future), **kwargs)
        except:
            return np.nan
    
    def _get_vars(self, data, future):
        X = None if self.X is None else self.X(data[future])
        y = None if self.y is None else self.y(data[future])
        return y, X
    
    def _fit(self, y, X, **kwargs):
        if self.model is None:
            self.model = pm.auto_arima(y, **kwargs)
        else:
            # self.model.add_new_observations(y.iloc[-1])
            self.model.fit(y.tail(20))
            # self.model.update(y.iloc[-1])
    
    def _predict(self, y, X, **kwargs):
        self._fit(y, X, **kwargs)
        y_pred = self.model.predict(n_periods=1)[0]
        y_pred = y_pred if self.y_pred is None else self.y_pred(y, y_pred)
        return y_pred

class Arima(ArimaWrapper):
    SAVED_DIR = 'saved_models/numeric/arima'
    
    def __init__(self, y_var='CLOSE', forecast='price'):
        super().__init__(
            y = self.transform_predictor,
            y_pred = self.transform_response,
        )
        self.SAVED_DIR = f'{self.SAVED_DIR}/{y_var}/{forecast}'
        self.y_var = y_var
        self.forecast = forecast
    
    def transform_predictor(self, data):
        """Returns the last 50 close prices in a train split"""
        rows = data.index < datetime(2021, 1, 1)
        cols = self.y_var
        y = data.loc[rows, cols]
        return y

    def transform_response(self, y, y_pred):
        if self.forecast == 'returns':
            return y_pred - y.iloc[-1]
        elif self.forecast == 'percent':
            return y_pred / y.iloc[-1]
        else:
            return y_pred


# Deprecated hopefully    
class ArimaRaw(ArimaWrapper):
    SAVED_DIR = 'saved_models/numeric/arimaraw'
    
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
