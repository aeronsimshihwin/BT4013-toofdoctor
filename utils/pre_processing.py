import math
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from typing import Callable, Mapping

def linearize(data: pd.DataFrame, old_var: str, new_var: str):
    data[new_var] = np.log(data[old_var])
    return data

def detrend(data: pd.DataFrame, old_var: str, new_var: str):
    data[new_var] = data[old_var].diff().fillna(0)
    return data

def add_stationary(data: pd.DataFrame, old_col, new_col):
    """Adds stationary close prices to the standardized dataset"""
    var = data[old_col]
    var = np.log(var) # Linearize
    var = var.diff() # Detrend      
    var = var.fillna(0)
    data[new_col] = var
    return data

def perc_change(var: pd.Series, periods:int=1, shift:int=1, offset:bool=False):
    if offset:
        return var.pct_change(periods=periods).shift(shift, freq=DateOffset(days=1))
    return var.pct_change(periods=periods).shift(shift)

def difference(var:pd.Series, periods:int=1, shift:int=1, offset:bool=False):
    if offset:
        return var.diff(periods=periods).shift(shift, freq=DateOffset(days=1))
    return var.diff(periods=periods).shift(shift)

def shift(var:pd.Series, shift:int=1, offset:bool=False, **kwargs):
    if offset:
        return var.shift(shift, freq=DateOffset(days=1))
    return var.shift(shift)

def generate_X_df(vars, trans, periods=[], shifts=[], offset:bool=False):
    periods = periods + [1] * (len(vars) - len(periods))
    shifts = shifts + [1] * (len(vars) - len(shifts))
    output = []
    for i in range(len(vars)):
        output.append(VAR_TRANSFORMATIONS[trans[i]](vars[i], periods=periods[i], offset=offset))
    return pd.concat(output, axis=1).dropna(axis=0)

def generate_y_cat(CLOSE):
    close_diff = difference(CLOSE, periods=1, shift=0)
    long_short = close_diff.apply(lambda x: np.nan if math.isnan(x) else 1 if x > 0 else -1)
    return long_short

def generate_y_diff(CLOSE):
    return difference(CLOSE, periods=1, shift=0)

def generate_y_perc(CLOSE):
    return perc_change(CLOSE, periods=1, shift=0)


VAR_TRANSFORMATIONS: Mapping[str, Callable] = {
    'perc': perc_change,
    'diff': difference,
    'shift': shift
}

Y_TRANSFORMATIONS: Mapping[str, Callable] = {
    'cat': generate_y_cat,
    'diff': generate_y_diff,
    'perc': generate_y_perc
}