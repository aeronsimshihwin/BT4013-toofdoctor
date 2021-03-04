from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pmdarima as pm
from tqdm import tqdm

import utils

SAVE_DIR = 'models/numeric/arima'
class ArimaWrapper:
    def __init__(self, models={}, y=None, X=None):
        self.models = models
        self.get_y = y # Extracts y from data[future]
        self.get_X = X # Extracts X from data[future]
        
    def save(self, save_dir):
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        for future in utils.futuresList:
            filepath = path / f'{future}.p'
            with filepath.open('wb') as f:
                pickle.dump(self.models[future], f)
            
    def load(self, save_dir):
        self.models = dict()
        for future in utils.futuresList:
            with open(f'{save_dir}/{future}.p', 'rb') as f:
                self.models[future] = pickle.load(f)
    
    def fit_predict(self, data, trace=None):
        y_diff = []        
        for future in tqdm(utils.futuresList):
            if future in data:
                X = None if self.get_X is None else self.get_X(data[future])
                y = None if self.get_y is None else self.get_y(data[future])
                y_last = y.iloc[-1]
            else:
                y_diff.append(np.nan)
                continue
            
            if future in self.models:
                self.models[future].update(y_last)
            else:
                self.models[future] = pm.auto_arima(y, trace=trace)
            
            y_pred = self.models[future].predict(n_periods=1)[0]
            y_diff.append(y_pred - y_last)                
                
        return pd.Series(y_diff, index=utils.futuresList)

def model(OPEN, HIGH, LOW, CLOSE, VOL, **kwargs):    
    predictions = CLOSE[utils.futuresList].apply(
        lambda x: pm.auto_arima(x.tail(40).dropna(), error_action='ignore').predict(n_periods=1)[0])
    
    current_value = CLOSE[utils.futuresList].tail(1).squeeze()
    price_diff = predictions - current_value
    return price_diff