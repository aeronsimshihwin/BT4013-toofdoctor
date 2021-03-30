import numpy as np
import pandas as pd
import utils

class fourCandleHammerWrapper:
    SAVED_DIR = "saved_models/categorical/fourCandleHammer"
    def __init__(self, model_params=None, y=None, X=None):
        self.model_params = model_params
        self.get_y = y # Extracts y from data[future]
        self.get_X = X # Extracts X from data[future]
        
    # Don't require fit function.
    # def fit(self, data, future): 
    #     X = None if self.get_X is None else self.get_X(data[future])
    #     y = None if self.get_y is None else self.get_y(data[future])

    def predict(self, data, future):
        data_df = data[future]
        # slice only relevant data
        X = data_df[self.get_X]

        strategy_df = utils.fourCandleHammer(X, self.model_params['N'], self.model_params['highFactor'], self.model_params['lowFactor'], future, self.model_params['macro_analysis'])
        y_pred = strategy_df[self.get_y][-1]
        return y_pred