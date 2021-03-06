import numpy as np
import pandas as pd
import math

import utils

SAVED_DIR = "/saved_models/categorical/fourCandleHammer"

class fourCandleHammerWrapper:
    # Reference: https://tradingstrategyguides.com/technical-analysis-strategy/
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
            
            close_ex5days = close[-24:-4]
            close_last5days = close[-5:]
            
            EMA13 = utils.EMA(pd.Series(close), 13)
            EMA26 = utils.EMA(pd.Series(close), 26)

            twentyDayNewHigh = close_ex5days[-1] >= 0.95 * max(close_ex5days)
            twentyDayNewLow = close_ex5days[-1] <= 1.05 * min(close_ex5days)
            if EMA13[-1] > EMA26[-1] and EMA13[-2] <= EMA26[-2]: # Uptrend
                if twentyDayNewHigh: # 1) Market made a 20-day new high
                    # 2) Identify 4 days pullback that goes against prevailing trend (4 consecutive days retracement)
                    if (close_last5days[0] > close_last5days[1]):
                        if (close_last5days[1] > close_last5days[2]):
                            if (close_last5days[2] > close_last5days[3]):
                                # 3) The latest closing price needs to be above the closing price (1 day before)
                                if close_last5days[4] > close_last5days[3]:
                                    y_diff = 1
            elif EMA13[-1] < EMA26[-1] and EMA13[-2] >= EMA26[-2]: # Downtrend
                if twentyDayNewLow: # 1) Market made a 20-day new low
                    # 2) Identify 4 days pullback that goes against prevailing trend (4 consecutive days retracement)
                    if (close_last5days[0] < close_last5days[1]):
                        if (close_last5days[1] < close_last5days[2]):
                            if (close_last5days[2] < close_last5days[3]):
                                # 3) The latest closing price needs to be below the closing price (1 day before)
                                if close_last5days[4] < close_last5days[3]:
                                    y_diff = -1
        return y_diff