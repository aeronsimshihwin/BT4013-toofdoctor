import numpy as np
import pandas as pd

def basic_strategy(sig, mag):
    return sig * mag

def fixed_threshold_strategy(sig, mag, thres):
    '''
    only invest in stocks with magnitudes that pass a specific threshold
    '''
    mag_mask = (mag > thres).astype("int")
    return mag_mask * mag * sig

def perc_threshold_strategy(sig, mag, thres):
    '''
    only invest in stocks with magnitudes above the threshold percentile of 
    magnitudes
    '''
    mag_percentile = mag.rank(method="average", pct=True)
    mag_mask = (mag_percentile > thres).astype("int")
    return mag_mask * mag * sig