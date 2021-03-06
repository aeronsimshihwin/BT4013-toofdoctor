import numpy as np
import pandas as pd
import math

import utils
from model_validation import walk_forward

SAVED_DIR = "/saved_models/categorical/swingSetup"

class swingSetupWrapper:
    def __init__(self, model=None, y=None, X=None):
        self.model = model
        self.get_y = y # Extracts y from data[future]
        self.get_X = X # Extracts X from data[future]
    
    def fit(self, data, future): 
        X = None if self.get_X is None else self.get_X(data[future])
        y = None if self.get_y is None else self.get_y(data[future])
        # Don't think this is needed for technical indicators

    def predict(self, data, future):
        y_diff = np.nan # Default return value
        if future in data:
            close = data['CLOSE']
            high = data['HIGH']
            low = data['LOW']

            SMA20 = SMA(pd.Series(close), 20)
            SMA40 = SMA(pd.Series(close), 40)
            CCI_indicator = CCI(pd.Series(high), pd.Series(low), pd.Series(close), 20)
            setup_time_index = -1
            if (gradient(SMA20) > 0 and gradient(SMA40) > 0): # Sloping up MAs
                for i in range(len(SMA20)-2, len(SMA20)-7, -1):
                    if (SMA20[i] > SMA40[i]): # SMA20 above SMA40
                        if (CCI_indicator[i] < -100): # CCI < -100, indicates price below avg
                            if (low[i] <= SMA20[i]): # low price touches or goes below SMA20
                                if (close[i] > SMA40[i]): # closing price goes above SMA40
                                    trigger_price = high[i] * 1.002
                                    if low[-1] >= trigger_price:
                                        y_diff = 1
            elif (gradient(SMA20) < 0 and gradient(SMA40) < 0): # Sloping down MAs
                for i in range(len(SMA20)-2, len(SMA20)-7, -1):
                    if (SMA20[i] < SMA40[i]): # SMA20 below SMA40
                        if (CCI_indicator[i] > 100): # CCI < -100, indicates price above avg
                            if (high[i] >= SMA20[i]): # high price touches or goes above SMA20
                                if (close[i] < SMA40[i]): # closing price goes below SMA40
                                    trigger_price = low[i] * 0.998
                                    if high[-1] <= trigger_price:
                                        y_diff = -1
        return y_diff