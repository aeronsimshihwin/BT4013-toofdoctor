import numpy as np
import pandas as pd

import utils

SAVED_DIR = "/saved_models/categorical/fourCandleHammer"

class fourCandleHammerWrapper:
    def __init__(self, model=None, y=None, X=None):
        self.model = model
        self.get_y = y # Extracts y from data[future]
        self.get_X = X # Extracts X from data[future]
    
    def fit(self, data, future): 
        X = None if self.get_X is None else self.get_X(data[future])
        y = None if self.get_y is None else self.get_y(data[future])
        # Don't think this is needed for technical indicators

    def predict(self, data, future, **kwargs):
        utils.fourCandleHammer(data)
        #TODO