import numpy as np
import pandas as pd

import utils

SAVED_DIR = "/saved_models/categorical/emaStrategy"

class emaStrategyWrapper:
    # Reference: https://tradingstrategyguides.com/exponential-moving-average-strategy/#:~:text=Many%20traders%20use%20exponential%20moving,level%20to%20execute%20your%20trade.
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
            
            EMA20 = utils.EMA(pd.Series(close), 20)
            EMA50 = utils.EMA(pd.Series(close), 50)
            
            crossover_time_index = -1
            retest_time_index_start = -1
            retest_time_index_end = -1
            num_retests = 0

            if (utils.gradient(EMA20) > 0) and (utils.gradient(EMA50) > 0): # Uptrend
                for i in range(len(EMA20)-1, 0, -1):
                    if (EMA20[i] > EMA50[i]) and (EMA20[i-1] <= EMA50[i-1]): # EMA20 crosses EMA50
                        crossover_time_index = i
                        break
                if crossover_time_index != -1:
                    for j in range(crossover_time_index, len(close)):
                        if (close[j] > EMA20[j]) and (close[j] > EMA50[j]): # After crossover, price closed above EMA20 and EMA50.
                            retest_time_index_start = j
                            break
                    if retest_time_index_start != -1:
                        retested = False
                        for k in range(retest_time_index_start, len(close)-1):
                            if not retested:
                                if (close[k+1] < close[k]) and (close[k+1] <= EMA20[k+1]) and (close[k+1] >= EMA50[k+1]):
                                    retested = True
                                    retest_time_index_end = k+1
                                    num_retests += 1
                            if retested:
                                if (close[k+1] > close[k]) and (close[k+1] > EMA20[k+1]):
                                    retested = False
                        if len(close) - retest_time_index_end <= 3: y_diff = 1 # If the last retest was within 3 days, it signals a BUY.
                    
            elif (utils.gradient(EMA20) < 0) and (utils.gradient(EMA50) < 0): # downtrend
                for i in range(len(EMA20)-1, 0, -1):
                    if (EMA20[i] <= EMA50[i]) and (EMA20[i-1] > EMA50[i-1]): # EMA50 crosses EMA20
                        crossover_time_index = i
                        break
                if crossover_time_index != -1:
                    for j in range(crossover_time_index, len(close)):
                        if (close[j] < EMA20[j]) and (close[j] < EMA50[j]): # After crossover, price closed below EMA20 and EMA50.
                            retest_time_index_start = j
                            break
                    if retest_time_index_start != -1:
                        retested = False
                        for k in range(retest_time_index_start, len(close)-1):
                            if not retested:
                                if (close[k+1] > close[k]) and (close[k+1] >= EMA20[k+1]) and (close[k+1] <= EMA50[k+1]):
                                    retested = True
                                    retest_time_index_end = k+1
                                    num_retests += 1
                                if retested:
                                    if (close[k+1] < close[k]) and (close[k+1] < EMA20[k+1]):
                                        retested = False
                        if len(close) - retest_time_index_end == 1: y_diff = -1 # If there was exactly 1 retest, it signals a SELL.
        return y_diff