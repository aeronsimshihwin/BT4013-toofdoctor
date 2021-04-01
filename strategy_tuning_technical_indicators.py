from models.categorical import (
    XGBWrapper,
    RFWrapper,
    LogRegWrapper
)

from models.numeric import (
    ArimaRaw, 
    ArimaLinear, 
    ArimaNoTrend, 
    ArimaLinearNoTrend,
)

from models.categorical import (
    XGBWrapper,
    RFWrapper,
    ArimaEnsemble,
    LogRegWrapper,
    fourCandleHammerWrapper,
    emaStrategyWrapper,
    swingSetupWrapper
)
from models.numeric import (
    Arima,
)
from strategy import (
    basic_strategy, 
    long_only,
    short_only,
    fixed_threshold_strategy, 
    perc_threshold_strategy,
    futures_only,
    futures_hold,
    cash_and_futures,
)

import utils
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# Must be imported last
from main import LOADED_MODELS

strategies = {
    'futures_only': futures_only,
    'futures_hold': futures_hold,
    'cash_and_futures': cash_and_futures
}
def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, USA_ADP, USA_EARN,\
    USA_HRS, USA_BOT, USA_BC, USA_BI, USA_CU, USA_CF, USA_CHJC, USA_CFNAI,\
    USA_CP, USA_CCR, USA_CPI, USA_CCPI, USA_CINF, USA_DFMI, USA_DUR,\
    USA_DURET, USA_EXPX, USA_EXVOL, USA_FRET, USA_FBI, USA_GBVL, USA_GPAY,\
    USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_CPIC, USA_CPICM,\
    USA_JBO, USA_LFPR, USA_LEI, USA_MPAY, USA_MP, USA_NAHB, USA_NLTTF,\
    USA_NFIB, USA_NFP, USA_NMPMI, USA_NPP, USA_EMPST, USA_PHS, USA_PFED,\
    USA_PP, USA_PPIC, USA_RSM, USA_RSY, USA_RSEA, USA_RFMI, USA_TVS, USA_UNR,\
    USA_WINV, exposure, equity, settings):
    ''' This system uses trend following techniques to allocate capital into the desired equities'''
    # Load standardized data
    date_index = pd.to_datetime(DATE, format='%Y%m%d')
    data = dict()

    # Data + preprocessing and indicators
    for i, future in enumerate(utils.futuresList):
        # Slice data by futures
        df = pd.DataFrame({
            'OPEN': OPEN[:, i], 
            'HIGH': HIGH[:, i],
            'LOW': LOW[:, i],
            'CLOSE': CLOSE[:, i],
            'VOL': VOL[:, i],
        }, index=date_index)
        
        # Replace nan and 0 values with previous day's data (ffill)
        df = df.replace(0, np.nan)
        df = df.fillna(method="ffill")

        # ARIMA: Velocity and acceleration terms for linearized data
        df = utils.linearize(df, old_var='CLOSE', new_var='CLOSE_LINEAR')
        df = utils.detrend(df, old_var='CLOSE_LINEAR', new_var='CLOSE_VELOCITY')
        df = utils.detrend(df, old_var='CLOSE_VELOCITY', new_var='CLOSE_ACCELERATION')
        
        # CATEGORICAL: Preprocessed features
        df = utils.percentage_diff(df, old_var='CLOSE', new_var='CLOSE_PCT')
        df = utils.diff(df, old_var='CLOSE', new_var='CLOSE_DIFF')
        df = utils.percentage_diff(df, old_var='CLOSE_LINEAR', new_var='CLOSE_LINEAR_PCT')

        df = utils.percentage_diff(df, old_var='VOL', new_var='VOL_PCT')
        df = utils.diff(df, old_var='VOL', new_var='VOL_DIFF')
        df = utils.linearize(df, old_var='VOL', new_var='VOL_LINEAR')
        df = utils.detrend(df, old_var='VOL_LINEAR', new_var='VOL_VELOCITY')
        df = utils.percentage_diff(df, old_var='VOL_LINEAR', new_var='VOL_LINEAR_PCT')

        # CATEGORICAL: Y var
        df = utils.long_short(df, old_var='CLOSE_DIFF', new_var='LONG_SHORT')

        # TECHNICAL INDICATORS
        df = utils.SMA(df, input='CLOSE', output='SMA20', periods=20)
        df = utils.EMA(df, input='CLOSE', output='EMA20', periods=20)
        df = utils.MACD(df, input='CLOSE', output='MACD')
        df = utils.RSI(df, input='CLOSE', output='RSI14', periods=14)
        df = utils.ATR(df, input=['HIGH', 'LOW', 'CLOSE'], output='ATR', periods=14)
        df = utils.VPT(df, input=['CLOSE', 'VOL'], output='VPT')
        df = utils.BBands(df, input='CLOSE', output=['BBANDS_HIGH', 'BBANDS_LOW'], periods=14)
        df = utils.CCI(df, input=['HIGH', 'LOW', 'CLOSE'], output='CCI', periods=20)

        df = df.replace(np.inf, np.nan)
        df = df.fillna(method="ffill")

        data[future] = df
    # Economic indicators
    indicators = pd.DataFrame(
        data = np.hstack([
            USA_ADP, USA_EARN,\
            USA_HRS, USA_BOT, USA_BC, USA_BI, USA_CU, USA_CF, USA_CHJC, USA_CFNAI,\
            USA_CP, USA_CCR, USA_CPI, USA_CCPI, USA_CINF, USA_DFMI, USA_DUR,\
            USA_DURET, USA_EXPX, USA_EXVOL, USA_FRET, USA_FBI, USA_GBVL, USA_GPAY,\
            USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_CPIC, USA_CPICM,\
            USA_JBO, USA_LFPR, USA_LEI, USA_MPAY, USA_MP, USA_NAHB, USA_NLTTF,\
            USA_NFIB, USA_NFP, USA_NMPMI, USA_NPP, USA_EMPST, USA_PHS, USA_PFED,\
            USA_PP, USA_PPIC, USA_RSM, USA_RSY, USA_RSEA, USA_RFMI, USA_TVS, USA_UNR,\
            USA_WINV,
        ]),
        index = date_index,
        columns = utils.keys, 
    )
    indicators = indicators.fillna(method="ffill")
    indicators = utils.percentage_diff(indicators, 
        old_var = utils.keys, 
        new_var = [key+"_PCT" for key in utils.keys],
    )
    indicators = utils.diff(
        indicators, 
        old_var = utils.keys, 
        new_var = [key+"_DIFF" for key in utils.keys],
    )

    for future, df in data.items():
        future_df = data[future].join(indicators)
        future_df = future_df.fillna(method="ffill")
        future_df = future_df.fillna(0)
        data[future] = future_df   
    
    # Fit and predict
    prediction = pd.DataFrame(index=utils.futuresList)
    for (name, future), model in tqdm(settings['models'].items()):
        prediction.loc[future, name] = model.predict(data, future) # , threshold=settings['threshold']
    sign = utils.sign(prediction)
    magnitude = utils.magnitude(prediction)
    
    # Futures strategy (Allocate position based on predictions)
    model = prediction.columns[0] # Arbitrarily pick first model in case of multiple
    position = basic_strategy(sign[model], magnitude[model]) 
    
    # Cash-futures strategy
    if settings['strategy'] == 'futures_hold':
        position = strategies[settings['strategy']](position, settings['previous_position'])
    else:
        position = strategies[settings['strategy']](position)

    # Update persistent data across runs
    settings['sign'].append(sign)
    settings['magnitude'].append(magnitude)
    settings['previous_position'] = position

    # Yay!
    return position, settings

def mySettings():
    ''' Define your trading system settings here '''
    settings= {}
    settings['markets']  = utils.futuresAllList
    settings['beginInSample'] = '20181020'
    settings['endInSample'] = '20201231'
    settings['lookback']= 504
    settings['budget']= 10**6
    settings['slippage']= 0.05

    # Stuff to persist
    settings['models'] = LOADED_MODELS
    
    with open('utils/strategy_tuning.txt','r') as f:
        strategy = f.readline()
        
    settings['strategy'] = strategy
    settings['sign'] = []
    settings['magnitude'] = []
    settings['previous_position'] = np.array([0] * len(utils.futuresList) + [1,])

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    
    model = 'fourCandleHammer'
    sharpe_results = []
    strategy_results = []

    for strategy in ['futures_only', 'futures_hold', 'cash_and_futures']:
        # retrieve sharpe
        results = quantiacsToolbox.runts(__file__, plotEquity=False)
        sharpe = results["stats"]["sharpe"]
        sharpe_results.append(sharpe)
        strategy_results.append(strategy)
    
    # save results
    results_df = pd.DataFrame({'strategy': strategy_results, 'sharpe': sharpe_results})
    best_result_df = results_df.loc[results_df["sharpe"] == max(results_df["sharpe"])].reset_index(drop=True)
    print(best_result_df)
    results_df.to_csv(f'model_metrics/strategy_threshold/{model}.csv')