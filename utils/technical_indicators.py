import datetime
import functools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from ta.trend import SMAIndicator, EMAIndicator, MACD, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.volume import VolumePriceTrendIndicator

def SMA(close, periods):
  """
  @param: close A series object containing closing prices.
  @param: periods An integer for the number of periods to average over.
  Returns a list of Simple Moving Average values.
  Function reference: https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.trend.SMAIndicator
  Info: https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/simple-moving-average-sma/
  """
  return SMAIndicator(close, periods).sma_indicator().tolist()

def EMA(close, periods):
  """
  @param: close A series object containing closing prices.
  @param: periods An integer for the number of periods to average over.
  Returns a list of Exponential Moving Average values. More weight is placed on recent prices. 
  The shorter the period, the more weight will be placed on most recent price.
  Function reference: https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.trend.EMAIndicator
  Info: https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/exponential-ema/
  """
  return EMAIndicator(close, periods).ema_indicator().tolist()

def MACD(close, fast=12, slow=26):
  """
  @param: close A series object containing closing prices.
  @param: fast An integer for the number of days to average over for Fast Moving Average. Default value set to 12.
  @param: slow An integer for the number of days to average over for Slow Moving Average. Default value set to 26.
  Returns a list of values that represent the difference between Fast Moving Average and Slow Moving Average.
  Info: https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/moving-average-convergence-divergence-macd/
  """
  res = []
  fastMA = EMA(close, fast)
  slowMA = EMA(close, slow)
  for a, b in zip(fastMA, slowMA):
    if b is not None:
      res.append(a-b)
  return res

def RSI(close, periods=14):
  """
  @param: close A series object containing closing prices.
  @param: periods An integer for the number of periods to average over. Default set to 14.
  Returns a list of values between 0 and 100, where values >= 70 indicates overbought/overvalued and values <= 30 indicates oversold/undervalued.
  Function reference: https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.momentum.RSIIndicator
  Info: https://www.investopedia.com/terms/r/rsi.asp
  """
  return RSIIndicator(close, periods).rsi().tolist()

def ATR(high, low, close, periods=14):
  """
  @param: high A series object containing high prices.
  @param: low A series object containing low prices.
  @param: close A series object containing closing prices.
  @param: periods An integer for the number of periods to average over. Default set to 14.
  Returns a list of values that represent the size of the price range for that period.
  Function reference: https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.volatility.AverageTrueRange
  Info: https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/average-true-range-atr/
  """
  return AverageTrueRange(high, low, close, periods).average_true_range().tolist()

def VPT(close, volume):
  """
  @param: close A series object containing closing prices.
  @param: volume A series object containing volume.
  Returns a list of percentage changes in share price trend and current volume
  Function reference: https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.volume.VolumePriceTrendIndicator
  Info: https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/price-volume-trend-pvt/
  """
  return VolumePriceTrendIndicator(close, volume).volume_price_trend().tolist()

def BBands(close, periods):
  """
  @param: close A series object containing closing prices.
  @param: periods An integer for the number of periods to average over.
  Returns a tuple containing 2 lists. First is the upper band, second is the lower band.
  Function reference: https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.volatility.BollingerBands
  Info: https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/bollinger-band-bbands/
  """
  BBands_high = BollingerBands(close, periods).bollinger_hband_indicator().tolist()
  BBands_low = BollingerBands(close, periods).bollinger_lband_indicator().tolist()
  return (BBands_high, BBands_low)

def CCI(high, low, close, periods=20):
    """
    @param: high A series object containing high prices.
    @param: low A series object containing low prices.
    @param: close A series object containing closing prices.
    @param: periods An integer for the number of periods to average over.
    Returns a list of differences between price change and average price change.
    Function reference: https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.trend.CCIIndicator
    Info: https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/commodity-channel-index-cci/
    """
    return CCIIndicator(high, low, close, periods).cci().tolist()

def gradient(lst):
    for val in lst:
        if (not math.isnan(val)) and (val != 0):
            return (lst[-1] - val)/val

def fourCandleHammer(close, N=30, highFactor=0.95, lowFactor=1.05):
    """
    @param: close A series object containing closing prices.
    @param: N The number of most recent days to scan through and find max and min price. For tuning purposes.
    @param: highFactor A constant value that will be multiplied to the max price found over the n recent days. For tuning purposes.
    @param: lowFactor A constant value that will be multiplied to the min price found over the n recent days. For tuning purposes.
    Returns 1 if should long futures, -1 if should short futures, 0 if do nothing.
    Strategy reference: https://tradingstrategyguides.com/technical-analysis-strategy/
    Note: N must be greater than 26, else EMA26 will give nan and not comparable to EMA13.
    """
    
    num_data_points = close.size
    start_idx = 0
    end_idx = N+5-1
    output = [-100] * end_idx
    while end_idx != num_data_points:
        close_ex5days = close[start_idx:end_idx-5]
        close_last5days = close[end_idx-5:end_idx]
        
        EMA13 = EMA(pd.Series(close), 13)
        EMA26 = EMA(pd.Series(close), 26)

        NDayNewHigh = close_ex5days[-1] >= highFactor * max(close_ex5days)
        NDayNewLow = close_ex5days[-1] <= lowFactor * min(close_ex5days)
        if EMA13[-1] > EMA26[-1] and EMA13[-2] <= EMA26[-2]: # Uptrend
            if NDayNewHigh: # 1) Market made a N-days new high
                # 2) Identify 4 days pullback that goes against prevailing trend (4 consecutive days retracement)
                if (close_last5days[0] > close_last5days[1]):
                    if (close_last5days[1] > close_last5days[2]):
                        if (close_last5days[2] > close_last5days[3]):
                            # 3) The latest closing price needs to be above the closing price from 1 day ago.
                            if close_last5days[4] > close_last5days[3]:
                                output.append(1)
        elif EMA13[-1] < EMA26[-1] and EMA13[-2] >= EMA26[-2]: # Downtrend
            if NDayNewLow: # 1) Market made a N-days new low
                # 2) Identify 4 days pullback that goes against prevailing trend (4 consecutive days retracement)
                if (close_last5days[0] < close_last5days[1]):
                    if (close_last5days[1] < close_last5days[2]):
                        if (close_last5days[2] < close_last5days[3]):
                            # 3) The latest closing price needs to be below the closing price from 1 day ago.
                            if close_last5days[4] < close_last5days[3]:
                                output.append(-1)
        else:
            output.append(0) # no long/short
        start_idx += 1
        end_idx += 1
    return output

def ema_strategy(close, shortTermDays=20, longTermDays=50):
    """
    @param: close A series object containing closing prices.
    @param: shortTermDays The number of days to calculate the short term EMA. For tuning purposes.
    @param: longTermDays The number of days to calculate the long term EMA. For tuning purposes.
    Returns 1 if should long futures, -1 if should short futures, 0 if do nothing.
    Strategy reference: https://tradingstrategyguides.com/exponential-moving-average-strategy/#:~:text=Many%20traders%20use%20exponential%20moving,level%20to%20execute%20your%20trade
    """
    EMAshort = EMA(pd.Series(close), shortTermDays)
    EMAlong = EMA(pd.Series(close), longTermDays)
    
    crossover_time_index = -1
    retest_time_index_start = -1
    retest_time_index_end = -1
    num_retests = 0

    if (gradient(EMAshort) > 0) and (gradient(EMAlong) > 0): # Uptrend
        for i in range(len(EMAshort)-1, 0, -1):
            if (EMAshort[i] > EMAlong[i]) and (EMAshort[i-1] <= EMAlong[i-1]): # EMAshort crosses EMAlong
                crossover_time_index = i
                break
        if crossover_time_index != -1:
            for j in range(crossover_time_index, len(close)):
                if (close[j] > EMAshort[j]) and (close[j] > EMAlong[j]): # After crossover, price closed above EMAshort and EMAlong.
                    retest_time_index_start = j
                    break
            if retest_time_index_start != -1:
                retested = False
                for k in range(retest_time_index_start, len(close)-1):
                    if not retested:
                        if (close[k+1] < close[k]) and (close[k+1] <= EMAshort[k+1]) and (close[k+1] >= EMAlong[k+1]):
                            retested = True
                            retest_time_index_end = k+1
                            num_retests += 1
                    if retested:
                        if (close[k+1] > close[k]) and (close[k+1] > EMAshort[k+1]):
                            retested = False
                if len(close) - retest_time_index_end <= 3: return 1 # If the last retest was within 3 days, it signals a LONG.
            
    elif (gradient(EMAshort) < 0) and (gradient(EMAlong) < 0): # downtrend
        for i in range(len(EMAshort)-1, 0, -1):
            if (EMAshort[i] <= EMAlong[i]) and (EMAshort[i-1] > EMAlong[i-1]): # EMAlong crosses EMAshort
                crossover_time_index = i
                break
        if crossover_time_index != -1:
            for j in range(crossover_time_index, len(close)):
                if (close[j] < EMAshort[j]) and (close[j] < EMAlong[j]): # After crossover, price closed below EMAshort and EMAlong.
                    retest_time_index_start = j
                    break
            if retest_time_index_start != -1:
                retested = False
                for k in range(retest_time_index_start, len(close)-1):
                    if not retested:
                        if (close[k+1] > close[k]) and (close[k+1] >= EMAshort[k+1]) and (close[k+1] <= EMAlong[k+1]):
                            retested = True
                            retest_time_index_end = k+1
                            num_retests += 1
                        if retested:
                            if (close[k+1] < close[k]) and (close[k+1] < EMAshort[k+1]):
                                retested = False
                if len(close) - retest_time_index_end == 1: return -1 # If there was exactly 1 retest, it signals a SHORT.
    return 0 # no long/short

def swing_setup(high, low, close, shortTermDays=20, longTermDays=40, highFactor=1.002, lowFactor=0.998):
    """
    @param: close A series object containing closing prices.
    @param: shortTermDays The number of days to calculate the short term SMA. For tuning purposes.
    @param: longTermDays The number of days to calculate the long term SMA. For tuning purposes.
    @param: highFactor A constant value that will be multiplied to the high price. For tuning purposes.
    @param: lowFactor A constant value that will be multiplied to the low price. For tuning purposes.
    Returns 1 if should long futures, -1 if should short futures, 0 if do nothing.
    """
    SMAshort = SMA(pd.Series(close), shortTermDays)
    SMAlong = SMA(pd.Series(close), longTermDays)
    CCI_indicator = CCI(pd.Series(high), pd.Series(low), pd.Series(close), 20)
    setup_time_index = -1
    if (gradient(SMAshort) > 0 and gradient(SMAlong) > 0): # Sloping up MAs
        for i in range(len(SMAshort)-2, len(SMAshort)-7, -1):
            if (SMAshort[i] > SMAlong[i]): # SMAshort above SMAlong
                if (CCI_indicator[i] < -100): # CCI < -100, indicates price below avg
                    if (low[i] <= SMAshort[i]): # low price touches or goes below SMAshort
                        if (close[i] > SMAlong[i]): # closing price goes above SMAlong
                            trigger_price = high[i] * highFactor
                            if low[-1] >= trigger_price:
                                return 1
    elif (gradient(SMAshort) < 0 and gradient(SMAlong) < 0): # Sloping down MAs
        for i in range(len(SMAshort)-2, len(SMAshort)-7, -1):
            if (SMAshort[i] < SMAlong[i]): # SMAshort below SMAlong
                if (CCI_indicator[i] > 100): # CCI > 100, indicates price above avg
                    if (high[i] >= SMAshort[i]): # high price touches or goes above SMAshort
                        if (close[i] < SMAlong[i]): # closing price goes below SMAlong
                            trigger_price = low[i] * lowFactor
                            if high[-1] <= trigger_price:
                                return -1
    return 0 # no long/short
