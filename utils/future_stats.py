import numpy as np
import pandas as pd
from quantiacsToolbox import stats

def market_stats(ret, long=True, short=True):
    # Unpack results
    mEquity = ret['marketEquity']
    exposure = ret['marketExposure']
    settings = ret['settings']
    returns = ret['returns']

    # Align index to data (not needed)
    # settings['markets'].insert(0, 'fundEquity')

    # From quantiacsToolbox.plotts (line 1000)
    records = []
    for index, market in enumerate(settings['markets']):

        # Prepare all the y-axes
        equityList = np.transpose(np.array(mEquity))

        Long = np.transpose(np.array(exposure))
        Long[Long < 0] = 0
        Long = Long[:, (settings['lookback'] - 2):-1]  # Market Exposure lagged by one day

        Short = - np.transpose(np.array(exposure))
        Short[Short < 0] = 0
        Short = Short[:, (settings['lookback'] - 2):-1]

        returnsList = np.transpose(np.array(returns))

        returnLong = np.transpose(np.array(exposure))
        returnLong[returnLong < 0] = 0
        returnLong[returnLong > 0] = 1
        returnLong = np.multiply(returnLong[:, (settings['lookback'] - 2):-1],
                                returnsList[:, (settings['lookback'] - 1):])  # y values for Long Only Equity Curve

        returnShort = - np.transpose(np.array(exposure))
        returnShort[returnShort < 0] = 0
        returnShort[returnShort > 0] = 1
        returnShort = np.multiply(returnShort[:, (settings['lookback'] - 2):-1],
                                returnsList[:, (settings['lookback'] - 1):])  # y values for Short Only Equity Curve

        # marketRet = np.transpose(np.array(marketReturns))
        # marketRet = marketRet[:, (settings['lookback'] - 1):]  # Ignore overall sharpe
        equityList = equityList[:, (settings['lookback'] - 1):]  # y values for all individual markets

        # Long & Short Selected
        if long and short:
            print(equityList)
            y_Equity = equityList[index - 1]

        # Long Selected
        elif long: 
            y_Equity = np.cumprod(1 + returnLong[index - 1])
        
        # Short Selected
        else:  
            y_Equity = np.cumprod(1 + returnShort[index - 1])

        records.append(stats(y_Equity))

    results = pd.DataFrame.from_records(records)
    results.index = settings['markets']
    results = results.drop(['CASH'])

    return results
