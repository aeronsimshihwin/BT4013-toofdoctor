import numpy as np
import pandas as pd
import pmdarima as pm

import utils

def model(OPEN, HIGH, LOW, CLOSE, VOL, **kwargs):    
    predictions = CLOSE[utils.futuresList].apply(
        lambda x: pm.auto_arima(x.tail(40).dropna(), error_action='ignore').predict(n_periods=1)[0])
    
    current_value = CLOSE[utils.futuresList].tail(1).squeeze()
    price_diff = predictions - current_value
    return price_diff