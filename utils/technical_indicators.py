import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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