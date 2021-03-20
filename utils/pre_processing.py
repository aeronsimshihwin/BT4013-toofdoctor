import math
import numpy as np
import pandas as pd
from datetime import date
from pandas.tseries.offsets import DateOffset
from typing import Callable, Mapping
import utils 

FUTURE_CORR_PATH = "utils/future_corr.csv"
PRICE_CORRELATIONS = pd.read_csv(FUTURE_CORR_PATH)
PRICE_CORRELATIONS = PRICE_CORRELATIONS.set_index("future")

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

def percentage_diff(data: pd.DataFrame, old_var, new_var, periods:int=1):
    data[new_var] = data[old_var].pct_change(periods=periods).fillna(0)
    data[data[new_var] == np.inf] = 0
    return data

def diff(data: pd.DataFrame, old_var, new_var, periods:int=1):
    data[new_var] = data[old_var].diff(periods=periods).fillna(0)
    return data

def shift(data: pd.DataFrame, old_var, new_var):
    data[new_var] = data[old_var].shift(periods).fillna(0)
    return data

def long_short(data: pd.DataFrame, old_var, new_var, periods:int=1):
    data[new_var] = data[old_var].apply(lambda x: np.nan if math.isnan(x) else 1 if x > 0 else -1)
    return data

def generate_X_vars(future):
    X_vars = ["MACD", "RSI14", "VPT"]
    future_corr = PRICE_CORRELATIONS.loc[future]
    if abs(future_corr["linear_corr"]) > abs(future_corr["exp_corr"]):
        X_vars.extend(["CLOSE_PCT", "VOL_PCT"])
    else:
        X_vars.extend(["CLOSE_LINEAR_PCT", "VOL_LINEAR_PCT"])
    return X_vars
    
    
def prepare_data(future):
    df = pd.read_csv(f"tickerData/{future}.txt", parse_dates=["DATE"])
    df.columns = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL', 'OI', 'P', 'R', 'RINFO']
    df = df.set_index("DATE")
    
    # replace nan and 0 values with previous day ffill
    df = df.replace(0, np.nan)
    df = df.fillna(method="ffill")
    
    # ARIMA: Velocity and acceleration terms for linearized data
    df = linearize(df, old_var='CLOSE', new_var='CLOSE_LINEAR')
    df = detrend(df, old_var='CLOSE_LINEAR', new_var='CLOSE_VELOCITY')
    df = detrend(df, old_var='CLOSE_VELOCITY', new_var='CLOSE_ACCELERATION')

    # CATEGORICAL: Preprocessed features
    df = percentage_diff(df, old_var='CLOSE', new_var='CLOSE_PCT')
    df = diff(df, old_var='CLOSE', new_var='CLOSE_DIFF')
    df = percentage_diff(df, old_var='CLOSE_LINEAR', new_var='CLOSE_LINEAR_PCT')

    df = percentage_diff(df, old_var='VOL', new_var='VOL_PCT')
    df = diff(df, old_var='VOL', new_var='VOL_DIFF')
    df = linearize(df, old_var='VOL', new_var='VOL_LINEAR')
    df = detrend(df, old_var='VOL_LINEAR', new_var='VOL_VELOCITY')
    df = percentage_diff(df, old_var='VOL_LINEAR', new_var='VOL_LINEAR_PCT')
    
    # CATEGORICAL: y variable (long/short)
    df = long_short(df, old_var='CLOSE_DIFF', new_var='LONG_SHORT')

    # TECHNICAL INDICATORS
    df = utils.SMA(df, input='CLOSE', output='SMA20', periods=20)
    df = utils.EMA(df, input='CLOSE', output='EMA20', periods=20)
    df = utils.MACD(df, input='CLOSE', output='MACD')
    df = utils.RSI(df, input='CLOSE', output='RSI14', periods=14)
    df = utils.ATR(df, input=['HIGH', 'LOW', 'CLOSE'], output='ATR', periods=14)
    df = utils.VPT(df, input=['CLOSE', 'VOL'], output='VPT')
    df = utils.BBands(df, input='CLOSE', output=['BBANDS_HIGH','BBANDS_LOW'], periods=14)
    df = utils.CCI(df, input=['HIGH', 'LOW', 'CLOSE'], output='CCI', periods=20)

    for key in utils.keys:
        key_df = pd.read_csv(f"tickerData/{key}.txt", parse_dates=["DATE"])
        key_df = key_df.set_index("DATE")
        df[key] = key_df["CLOSE"]
        # forward fill to replace nonexistent values with previous values
        df[key] = df[key].fillna(method="ffill")
        # backfill to fill first 2 nan values
        df = percentage_diff(df, old_var=key, new_var=key+"_PCT")
        df = diff(df, old_var=key, new_var=key+"_DIFF")
        
    return df

def generate_X_y(df:pd.DataFrame, X_vars=[], y_var="LONG_SHORT", start_date=date(2010, 1, 1)):
    X_df = df[X_vars].shift(1).dropna()[start_date:]
    y_df = df[y_var].dropna()[start_date:]

    # retrieve only common index
    common_index = X_df.index.intersection(y_df.index)
    X_df = X_df[X_df.index.isin(common_index)]
    y_df = y_df[y_df.index.isin(common_index)]
    
    return X_df, y_df