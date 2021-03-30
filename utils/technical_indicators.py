import datetime
import functools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import utils

from ta.trend import SMAIndicator, EMAIndicator, MACD, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.volume import VolumePriceTrendIndicator

FUTURE_INDUSTRY_PATH = "utils/future_industry_mapping.csv"
FUTURE_INDUSTRY = pd.read_csv(FUTURE_INDUSTRY_PATH)
FUTURE_INDUSTRY = FUTURE_INDUSTRY.set_index("Ticker")

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
#   close_prev = close.shift(1)
#   vpt = volume * (close - close_prev) / close_prev
#   vpt[0] = 0
#   vpt_final = vpt.shift(1, fill_value=0) + vpt
  return VolumePriceTrendIndicator(close, volume, fillna=True).volume_price_trend().tolist() # vpt_final.tolist()

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

def fourCandleHammer(df, N, highFactor, lowFactor, futures, macro_analysis):
    """
    @param: df A Dataframe object containing relevant info of a futures.
    @param: N The number of most recent days to scan through and find max and min price. For tuning purposes.
    @param: highFactor A constant value that will be multiplied to the max price found over the n recent days. For tuning purposes.
    @param: lowFactor A constant value that will be multiplied to the min price found over the n recent days. For tuning purposes.
    @param: futures A string object to identify what futures it is.
    @param: macro_analysis A boolean object to indicate whether we are using macroeconomic indicator as our analysis.
    Returns dataframe, updated with long/short for each day.
    Strategy reference: https://tradingstrategyguides.com/technical-analysis-strategy/
    Note: N must be greater than 26, else EMA26 will give nan and not comparable to EMA13.
    """
    close = df['CLOSE']
    num_data_points = df.shape[0]
    # Strategy requires at least (N+5) days of prices to predict long/short for (N+6)th day.
    # [start index, end index] of data to look and to determine whether long/short with strategy.
    window_frame = [0, N+4] 

    # Strategy will not determine long/short for the first (N+5) days, since there is insufficient data.
    # Set this value to -100.
    long_short = [-100] * (N+5)

    # Strategy will continuously determine long/short starting from (N+6)th day till last day in data.
    while window_frame[1] < num_data_points-1:
        close_ex5days = close[window_frame[0]:window_frame[1]-4] # Series to identify N-day high/low.
        close_last5days = close[window_frame[1]-4:window_frame[1]+1] # Series to identify retracement.

        NDayNewHigh = close_ex5days[-1] >= highFactor * max(close_ex5days)
        NDayNewLow = close_ex5days[-1] <= lowFactor * min(close_ex5days)
        output = False
        if NDayNewHigh: # 1) Market made a N-days new high
            # 2) Identify 4 days pullback that goes against prevailing trend (4 consecutive days retracement)
            if (close_last5days[0] > close_last5days[1]):
                if (close_last5days[1] > close_last5days[2]):
                    if (close_last5days[2] > close_last5days[3]):
                        # 3) The latest closing price needs to be above the closing price from 1 day ago.
                        if close_last5days[4] > close_last5days[3]:
                            if macro_analysis == True:
                                df2 = df.iloc[window_frame[0]:window_frame[1]+1, :]
                                if analyse_macroeconomic_indicators(df2, futures, 1):
                                    long_short.append(1)
                                    output = True
                            else:
                                long_short.append(1)
                                output = True
        elif NDayNewLow: # 1) Market made a N-days new low
            # 2) Identify 4 days pullback that goes against prevailing trend (4 consecutive days retracement)
            if (close_last5days[0] < close_last5days[1]):
                if (close_last5days[1] < close_last5days[2]):
                    if (close_last5days[2] < close_last5days[3]):
                        # 3) The latest closing price needs to be below the closing price from 1 day ago.
                        if close_last5days[4] < close_last5days[3]:
                            if macro_analysis == True:
                                df2 = df.iloc[window_frame[0]:window_frame[1]+1, :]
                                if analyse_macroeconomic_indicators(df2, futures, -1):
                                    long_short.append(-1)
                                    output = True
                            else:
                                long_short.append(-1)
                                output = True
        if output == False:
            long_short.append(0) # no long/short
        window_frame[0] += 1
        window_frame[1] += 1

    df_with_long_short = df.copy()
    df_with_long_short['LONG_SHORT'] = long_short

    return df_with_long_short

def ema_strategy(df, shortTermDays, longTermDays, NDays, futures, macro_analysis):
    """
    @param: df A Dataframe object containing relevant info of a futures.
    @param: shortTermDays The number of days to calculate the short term EMA. For tuning purposes.
    @param: longTermDays The number of days to calculate the long term EMA. For tuning purposes.
    @param: NDays The number of days to compute trend. For tuning purposes.
    @param: futures A string object to identify what futures it is.
    @param: macro_analysis A boolean object to indicate whether we are using macroeconomic indicator as our analysis.
    Returns dataframe, updated with long/short for each day.
    Strategy reference: https://tradingstrategyguides.com/exponential-moving-average-strategy/#:~:text=Many%20traders%20use%20exponential%20moving,level%20to%20execute%20your%20trade
    """
    close_temp = df['CLOSE']
    num_data_points = df.shape[0]

    # Strategy requires at least (NDays) of prices to predict long/short for (NDays+1)th day.
    # [window_start, window_start+NDays] of data to look and to determine whether long/short with strategy.
    window_start = 0

    # Strategy will not determine long/short for the first (NDays) days, since there is insufficient data.
    # Set this value to -100.
    long_short = [-100] * NDays
    start_idx = NDays-1
    # Strategy will continuously determine long/short starting from (NDays+1)th day till last day in data.
    while start_idx < num_data_points-1:
        close = close_temp[window_start:window_start+NDays+1]
        EMAshort = EMA(pd.Series(close), shortTermDays)
        EMAlong = EMA(pd.Series(close), longTermDays)
        
        crossover_time_index = -1
        retest_time_index_start = -1
        retest_time_index_end = -1
        num_retests = 0
        output = False

        if (gradient(EMAshort) > 0) and (gradient(EMAlong) > 0): # Uptrend
            for i in range(len(EMAshort)-1, 0, -1):
                if (EMAshort[i] > EMAlong[i]) and (EMAshort[i-1] <= EMAlong[i-1]): # EMAshort crosses EMAlong
                    crossover_time_index = i
                    break
            if crossover_time_index != -1:
                for j in range(crossover_time_index, NDays):
                    if (close[j] > EMAshort[j]) and (close[j] > EMAlong[j]): # After crossover, price closed above EMAshort and EMAlong.
                        retest_time_index_start = j
                        break
                if retest_time_index_start != -1:
                    retested = False
                    for k in range(retest_time_index_start, NDays-1):
                        if not retested:
                            if (close[k+1] < close[k]) and (close[k+1] <= EMAshort[k+1]) and (close[k+1] >= EMAlong[k+1]):
                                retested = True
                                retest_time_index_end = k+1
                                num_retests += 1
                        if retested:
                            if (close[k+1] > close[k]) and (close[k+1] > EMAshort[k+1]):
                                retested = False
                    if num_retests >= 3: # If there were at least 3 retests, it signals a LONG.
                        if macro_analysis == True:
                            df2 = df.iloc[window_start:window_start+NDays+1, :]
                            if analyse_macroeconomic_indicators(df2, futures, 1):
                                long_short.append(1)
                                output = True
                        else:
                            long_short.append(1)
                            output = True
                
        elif (gradient(EMAshort) < 0) and (gradient(EMAlong) < 0): # downtrend
            for i in range(len(EMAshort)-1, 0, -1):
                if (EMAshort[i] <= EMAlong[i]) and (EMAshort[i-1] > EMAlong[i-1]): # EMAlong crosses EMAshort
                    crossover_time_index = i
                    break
            if crossover_time_index != -1:
                for j in range(crossover_time_index, NDays):
                    if (close[j] < EMAshort[j]) and (close[j] < EMAlong[j]): # After crossover, price closed below EMAshort and EMAlong.
                        retest_time_index_start = j
                        break
                if retest_time_index_start != -1:
                    retested = False
                    for k in range(retest_time_index_start, NDays-1):
                        if not retested:
                            if (close[k+1] > close[k]) and (close[k+1] >= EMAshort[k+1]) and (close[k+1] <= EMAlong[k+1]):
                                retested = True
                                retest_time_index_end = k+1
                                num_retests += 1
                            if retested:
                                if (close[k+1] < close[k]) and (close[k+1] < EMAshort[k+1]):
                                    retested = False
                    if num_retests >= 1: # If there was exactly 1 retest, it signals a SHORT.
                        if macro_analysis == True:
                            df2 = df.iloc[window_start:window_start+NDays+1, :]
                            if analyse_macroeconomic_indicators(df2, futures, -1):
                                long_short.append(-1)
                                output = True
                        else:
                            long_short.append(-1)
                            output = True

        if output == False:
            long_short.append(0) # no long/short
        window_start += 1
        start_idx += 1

    if len(long_short) != df.shape[0]: # Check
        print(len(long_short))
        print(df.shape[0])
        return False 

    df_with_long_short = df.copy()
    df_with_long_short['LONG_SHORT'] = long_short

    return df_with_long_short

def swing_setup(df, shortTermDays, longTermDays, NDays, futures, macro_analysis):
    """
    @param: df A Dataframe object containing relevant info of a futures.
    @param: shortTermDays The number of days to calculate the short term SMA. For tuning purposes.
    @param: longTermDays The number of days to calculate the long term SMA. For tuning purposes.
    @param: NDays The number of days to compute trend. For tuning purposes.
    @param: futures A string object to identify what futures it is.
    @param: macro_analysis A boolean object to indicate whether we are using macroeconomic indicator as our analysis.
    Returns dataframe, updated with long/short for each day.
    """
    close_temp = df['CLOSE']
    high_temp = df['HIGH']
    low_temp = df['LOW']

    num_data_points = df.shape[0]

    # Strategy requires at least (NDays) of prices to predict long/short for (NDays+1)th day.
    # [window_start, window_start+NDays] of data to look and to determine whether long/short with strategy.
    window_start = 0

    # Strategy will not determine long/short for the first (NDays) days, since there is insufficient data.
    # Set this value to -100.
    long_short = [-100] * NDays
    start_idx = NDays-1
    counter = 0
    # Strategy will continuously determine long/short starting from (NDays+1)th day till last day in data.
    while start_idx < num_data_points-1:
        close = close_temp[window_start:window_start+NDays+1]
        high = high_temp[window_start:window_start+NDays+1]
        low = low_temp[window_start:window_start+NDays+1]

        SMAshort = SMA(pd.Series(close), shortTermDays)
        SMAlong = SMA(pd.Series(close), longTermDays)
        CCI_indicator = CCI(pd.Series(high), pd.Series(low), pd.Series(close), 20)
        output = False
        
        if (gradient(SMAshort) > 0) and (gradient(SMAlong) > 0): # Sloping up MAs
            for i in range(len(SMAshort)-2, len(SMAshort)-7, -1):
                if (SMAshort[i] > SMAlong[i]): # SMAshort above SMAlong
                    if (CCI_indicator[i] < -100): # CCI < -100, indicates price below avg
                        if (low[i] <= SMAshort[i]): # low price touches or goes below SMAshort
                            if (close[i] > SMAlong[i]): # closing price goes above SMAlong
                                trigger_price = high[i] * 1.002
                                if low[-1] >= trigger_price:
                                    if macro_analysis == True:
                                        df2 = df.iloc[window_start:window_start+NDays+1, :]
                                        if analyse_macroeconomic_indicators(df2, futures, 1):
                                            long_short.append(1)
                                            output = True
                                            break
                                    else:
                                        long_short.append(1)
                                        output = True
                                        break
        elif (gradient(SMAshort) < 0) and (gradient(SMAlong) < 0): # Sloping down MAs
            for i in range(len(SMAshort)-2, len(SMAshort)-7, -1):
                if (SMAshort[i] < SMAlong[i]): # SMAshort below SMAlong
                    if (CCI_indicator[i] > 100): # CCI > 100, indicates price above avg
                        if (high[i] >= SMAshort[i]): # high price touches or goes above SMAshort
                            if (close[i] < SMAlong[i]): # closing price goes below SMAlong
                                trigger_price = low[i] * 0.998
                                if high[-1] <= trigger_price:
                                    if macro_analysis == True:
                                        df2 = df.iloc[window_start:window_start+NDays+1, :]
                                        if analyse_macroeconomic_indicators(df2, futures, -1):
                                            long_short.append(-1)
                                            output = True
                                            break
                                    else:
                                        long_short.append(-1)
                                        output = True
                                        break

        if output == False:
            long_short.append(0)
        window_start += 1
        start_idx += 1
        
    if len(long_short) != df.shape[0]: # Check
        print(len(long_short))
        print(df.shape[0])
        return False 

    df_with_long_short = df.copy()
    df_with_long_short['LONG_SHORT'] = long_short

    return df_with_long_short

def analyse_macroeconomic_indicators(df, futures, longOrShort):
    '''
    @param: df A Dataframe object containing relevant info of a futures.
    @param: futures A string object to indicate the futures.
    @param: longOrShort An integer where 1 indicates Long and -1 indicates Short.
    Returns True if macroeconomic indicators support longOrShort position, else False.
    Exception: Returns True when futures is not from United States, as no macroeconmic indicators available to analyse.
    '''
    futures_info = FUTURE_INDUSTRY.loc[futures]
    isUnitedStates = futures_info['UnitedStates'] == 1
    if not isUnitedStates: return True 
    
    futures_ind = futures_info['Type']

    # Long
    if longOrShort == 1:
        if futures_ind == "Agriculture":
            pp_lst = df['USA_PP'].tolist()
            nfp_lst = df['USA_NFP'].tolist()
            ccpi_lst = df['USA_CCPI'].tolist()
            cfnai_lst = df['USA_CFNAI'].tolist()
            ## USA_PP : Compute percentage change over a period. If positive (negative), price of domestic products - agriculture increase (decrease).
            pp_flag = gradient(pp_lst) > 0

            ## USA_NFP : Compute percentage change over a period. If positive (negative), more workers in the US, more consumers have purchasing power.
            nfp_flag = gradient(nfp_lst) > 0

            ## USA_CCPI : Compute percentage change over a period, proxy for core inflation rate. If greater than 2%, considered high inflation. 
            # Reference : https://www.thebalance.com/core-inflation-rate-3305918#:~:text=3-,What%20Is%20Core%20Inflation%3F,reading%20of%20underlying%20inflation%20trends.
            ccpi_flag = gradient(ccpi_lst) > 0.02

            ## USA_CFNAI : A positive (negative) index reading corresponds to growth above (below) trend.
            # Reference : https://www.chicagofed.org/publications/cfnai/index
            cfnai_flag = cfnai_lst[-1] > 0

            res = (pp_flag, nfp_flag, ccpi_flag, cfnai_flag)
            if sum(res) >= 2: return True # 2 or more indicators (out of 4) analyzed to be the same result, we take that.

        elif futures_ind == "Energy":
            nfp_lst = df['USA_NFP'].tolist()
            ccpi_lst = df['USA_CCPI'].tolist()
            cfnai_lst = df['USA_CFNAI'].tolist()

            nfp_flag = gradient(nfp_lst) > 0

            ccpi_flag = gradient(ccpi_lst) > 0.02

            cfnai_flag = cfnai_lst[-1] > 0

            res = (nfp_flag, ccpi_flag, cfnai_flag)
            if sum(res) >= 2: return True # 2 or more indicators (out of 3) analyzed to be the same result, we take that.

        elif futures_ind == "Currency":
            nfp_lst = df['USA_NFP'].tolist()
            ipmom_lst = df['USA_IPMOM'].tolist()
            cpi_lst = df['USA_CPI'].tolist()
            unr_lst = df['USA_UNR'].tolist()

            nfp_flag = gradient(nfp_lst) > 0

            ## USA_IPMOM : Compute percentage change over a period. If positive (negative), good (bad) economic health.
            ipmom_flag = gradient(ipmom_lst) > 0

            ## USA_CPI : Compute percentage change over a period, proxy for inflation rate. If greater than 1.5%, considered high inflation. 
            # Reference : https://www.researchgate.net/publication/311413446_Inflation_and_Growth_An_Estimate_of_the_Threshold_Level_of_Inflation_in_the_US#:~:text=The%20model%20suggests%20that%20the,real%20GDP%20growth%20is%20ambiguous.
            cpi_flag = gradient(cpi_lst) > 0.15
            
            ## USA_UNR : Compute percentage change over a period. If positive (negative), bad (good) for economy.
            unr_flag = gradient(unr_lst) < 0

            res = (nfp_flag, ipmom_flag, cpi_flag, unr_flag)
            if sum(res) >= 2: return True # 2 or more indicators (out of 4) analyzed to be the same result, we take that.

        elif futures_ind == "Index":
            cfnai_lst = df['USA_CFNAI'].tolist()
            cfnai_flag = cfnai_lst[-1] > 0

            return cfnai_flag

        elif futures_ind == "Bond":
            cpi_lst = df['USA_CPI'].tolist()
            cfnai_lst = df['USA_CFNAI'].tolist()
            unr_lst = df['USA_UNR'].tolist()

            cpi_flag = gradient(cpi_lst) > 0.15

            cfnai_flag = cfnai_lst[-1] > 0

            unr_flag = gradient(unr_lst) < 0

            res = (cpi_flag, cfnai_flag, unr_flag)
            if sum(res) >= 2: return True # 2 or more indicators (out of 3) analyzed to be the same result, we take that.

        elif futures_ind == "Metal":
            nfp_lst = df['USA_NFP'].tolist()
            ccpi_lst = df['USA_CCPI'].tolist()
            cfnai_lst = df['USA_CFNAI'].tolist()

            nfp_flag = gradient(nfp_lst) > 0
            
            ccpi_flag = gradient(ccpi_lst) > 0.02

            cfnai_flag = cfnai_lst[-1] > 0

            res = (nfp_flag, ccpi_flag, cfnai_flag)
            if sum(res) >= 2: return True # 2 or more indicators (out of 3) analyzed to be the same result, we take that.

        elif futures_ind == "Interest Rate":
            lei_lst = df['USA_LEI'].tolist()
            cpicm_lst = df['USA_CPICM'].tolist()
            unr_lst = df['USA_UNR'].tolist()

            ## USA_LEI
            # Compute percentage change over a period. If positive (negative), interest rates rise (fall).
            lei_flag = gradient(lei_lst) > 0

            ## USA_CPICM
            # Compute percentage change over a period. If positive (negative), high (low) inflation and interest rates fall (rise).
            cpicm_flag = gradient(cpicm_lst) < 0

            unr_flag = gradient(unr_lst) < 0
        
            res = (lei_flag, cpicm_flag, unr_flag)
            if sum(res) >= 2: return True # 2 or more indicators (out of 3) analyzed to be the same result, we take that.

    # Short
    elif longOrShort == -1:
        if futures_ind == "Agriculture":
            pp_lst = df['USA_PP'].tolist()
            nfp_lst = df['USA_NFP'].tolist()
            ccpi_lst = df['USA_CCPI'].tolist()
            cfnai_lst = df['USA_CFNAI'].tolist()
            ## USA_PP : Compute percentage change over a period. If positive (negative), price of domestic products - agriculture increase (decrease).
            pp_flag = gradient(pp_lst) < 0

            ## USA_NFP : Compute percentage change over a period. If positive (negative), more workers in the US, more consumers have purchasing power.
            nfp_flag = gradient(nfp_lst) < 0

            ## USA_CCPI : Compute percentage change over a period, proxy for core inflation rate. If greater than 2%, considered high inflation. 
            # Reference : https://www.thebalance.com/core-inflation-rate-3305918#:~:text=3-,What%20Is%20Core%20Inflation%3F,reading%20of%20underlying%20inflation%20trends.
            ccpi_flag = gradient(ccpi_lst) < 0.02

            ## USA_CFNAI : A positive (negative) index reading corresponds to growth above (below) trend.
            # Reference : https://www.chicagofed.org/publications/cfnai/index
            cfnai_flag = cfnai_lst[-1] < 0

            res = (pp_flag, nfp_flag, ccpi_flag, cfnai_flag)
            if sum(res) >= 2: return True # 2 or more indicators (out of 4) analyzed to be the same result, we take that.

        elif futures_ind == "Energy":
            nfp_lst = df['USA_NFP'].tolist()
            ccpi_lst = df['USA_CCPI'].tolist()
            cfnai_lst = df['USA_CFNAI'].tolist()

            nfp_flag = gradient(nfp_lst) < 0

            ccpi_flag = gradient(ccpi_lst) < 0.02

            cfnai_flag = cfnai_lst[-1] < 0

            res = (nfp_flag, ccpi_flag, cfnai_flag)
            if sum(res) >= 2: return True # 2 or more indicators (out of 3) analyzed to be the same result, we take that.

        elif futures_ind == "Currency":
            nfp_lst = df['USA_NFP'].tolist()
            ipmom_lst = df['USA_IPMOM'].tolist()
            cpi_lst = df['USA_CPI'].tolist()
            unr_lst = df['USA_UNR'].tolist()

            nfp_flag = gradient(nfp_lst) < 0

            ## USA_IPMOM : Compute percentage change over a period. If positive (negative), good (bad) economic health.
            ipmom_flag = gradient(ipmom_lst) < 0

            ## USA_CPI : Compute percentage change over a period, proxy for inflation rate. If greater than 1.5%, considered high inflation. 
            # Reference : https://www.researchgate.net/publication/311413446_Inflation_and_Growth_An_Estimate_of_the_Threshold_Level_of_Inflation_in_the_US#:~:text=The%20model%20suggests%20that%20the,real%20GDP%20growth%20is%20ambiguous.
            cpi_flag = gradient(cpi_lst) < 0.15
            
            ## USA_UNR : Compute percentage change over a period. If positive (negative), bad (good) for economy.
            unr_flag = gradient(unr_lst) > 0

            res = (nfp_flag, ipmom_flag, cpi_flag, unr_flag)
            if sum(res) >= 2: return True # 2 or more indicators (out of 4) analyzed to be the same result, we take that.

        elif futures_ind == "Index":
            cfnai_lst = df['USA_CFNAI'].tolist()
            cfnai_flag = cfnai_lst[-1] < 0

            return cfnai_flag

        elif futures_ind == "Bond":
            cpi_lst = df['USA_CPI'].tolist()
            cfnai_lst = df['USA_CFNAI'].tolist()
            unr_lst = df['USA_UNR'].tolist()

            cpi_flag = gradient(cpi_lst) < 0.15

            cfnai_flag = cfnai_lst[-1] < 0

            unr_flag = gradient(unr_lst) > 0

            res = (cpi_flag, cfnai_flag, unr_flag)
            if sum(res) >= 2: return True # 2 or more indicators (out of 3) analyzed to be the same result, we take that.

        elif futures_ind == "Metal":
            nfp_lst = df['USA_NFP'].tolist()
            ccpi_lst = df['USA_CCPI'].tolist()
            cfnai_lst = df['USA_CFNAI'].tolist()

            nfp_flag = gradient(nfp_lst) < 0
            
            ccpi_flag = gradient(ccpi_lst) < 0.02

            cfnai_flag = cfnai_lst[-1] < 0

            res = (nfp_flag, ccpi_flag, cfnai_flag)
            if sum(res) >= 2: return True # 2 or more indicators (out of 3) analyzed to be the same result, we take that.

        elif futures_ind == "Interest Rate":
            lei_lst = df['USA_LEI'].tolist()
            cpicm_lst = df['USA_CPICM'].tolist()
            unr_lst = df['USA_UNR'].tolist()

            ## USA_LEI
            # Compute percentage change over a period. If positive (negative), interest rates rise (fall).
            lei_flag = gradient(lei_lst) < 0

            ## USA_CPICM
            # Compute percentage change over a period. If positive (negative), high (low) inflation and interest rates fall (rise).
            cpicm_flag = gradient(cpicm_lst) > 0

            unr_flag = gradient(unr_lst) > 0
        
            res = (lei_flag, cpicm_flag, unr_flag)
            if sum(res) >= 2: return True # 2 or more indicators (out of 3) analyzed to be the same result, we take that.

    return False