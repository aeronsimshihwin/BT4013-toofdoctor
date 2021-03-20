import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.categorical import (
    XGBWrapper,
    RFWrapper,
    ArimaEnsemble,
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
    cash_and_futures,
    strategies_eval
)
import utils

# Load saved models
SAVED_MODELS = {
    # 'rf': RFWrapper,
    # 'xgb': XGBWrapper,
    'arima+xgb': ArimaEnsemble,
}

LOADED_MODELS = {}
for name, model in SAVED_MODELS.items():
    print(f'loading {name} from {model.SAVED_DIR}...')
    for future in utils.futuresList:
        pickle_path = f'{model.SAVED_DIR}/{future}.p'
        try:
            with open(pickle_path, 'rb') as f:
                LOADED_MODELS[name, future] = pickle.load(f)
        except:
            raise FileNotFoundError(f'No saved {name} for {future}!')


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
        
        # ARIMA: Velocity and acceleration terms for linearized data
        df = utils.linearize(df, old_var='CLOSE', new_var='CLOSE_LINEAR')
        df = utils.detrend(df, old_var='CLOSE_LINEAR', new_var='CLOSE_VELOCITY')
        df = utils.detrend(df, old_var='CLOSE_VELOCITY', new_var='CLOSE_ACCELERATION')
        
        pass # Add technical_indicators as columns in each future dataframe
        pass # Add preprocessed features as columns in each future dataframe
        
        data[future] = df

    # Economic indicators
    keys = (
        'USA_ADP, USA_EARN,\
        USA_HRS, USA_BOT, USA_BC, USA_BI, USA_CU, USA_CF, USA_CHJC, USA_CFNAI,\
        USA_CP, USA_CCR, USA_CPI, USA_CCPI, USA_CINF, USA_DFMI, USA_DUR,\
        USA_DURET, USA_EXPX, USA_EXVOL, USA_FRET, USA_FBI, USA_GBVL, USA_GPAY,\
        USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_CPIC, USA_CPICM,\
        USA_JBO, USA_LFPR, USA_LEI, USA_MPAY, USA_MP, USA_NAHB, USA_NLTTF,\
        USA_NFIB, USA_NFP, USA_NMPMI, USA_NPP, USA_EMPST, USA_PHS, USA_PFED,\
        USA_PP, USA_PPIC, USA_RSM, USA_RSY, USA_RSEA, USA_RFMI, USA_TVS, USA_UNR,\
        USA_WINV'
    ).replace(' ', '').split(',')
    vals = (
        USA_ADP, USA_EARN,\
        USA_HRS, USA_BOT, USA_BC, USA_BI, USA_CU, USA_CF, USA_CHJC, USA_CFNAI,\
        USA_CP, USA_CCR, USA_CPI, USA_CCPI, USA_CINF, USA_DFMI, USA_DUR,\
        USA_DURET, USA_EXPX, USA_EXVOL, USA_FRET, USA_FBI, USA_GBVL, USA_GPAY,\
        USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_CPIC, USA_CPICM,\
        USA_JBO, USA_LFPR, USA_LEI, USA_MPAY, USA_MP, USA_NAHB, USA_NLTTF,\
        USA_NFIB, USA_NFP, USA_NMPMI, USA_NPP, USA_EMPST, USA_PHS, USA_PFED,\
        USA_PP, USA_PPIC, USA_RSM, USA_RSY, USA_RSEA, USA_RFMI, USA_TVS, USA_UNR,\
        USA_WINV,
    )
    for key, val in zip(keys, vals):
        data[key] = pd.DataFrame(
            data = val,
            index = date_index,
            columns = ['CLOSE'],
        )

    ### Technical indicator strategy output ###
    ### Added here temporarily. Aeron to get help from Mitch ###
    # nMarkets = CLOSE.shape[1]
    # position = np.zeros(nMarkets)
    # index = 0
    # for future in utils.futuresList:
    #     # load data
    #     df = pd.read_csv(f"tickerData/{future}.txt", parse_dates = ["DATE"])
    #     df.columns = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL', 'OI', 'P', 'R', 'RINFO']
    #     df = df.set_index("DATE")
    #     df = df[(df.VOL != 0) & (df.CLOSE != 0)]
    #     df = df.dropna(axis=0)

    #     # position[index+1] = utils.fourCandleHammer(df['CLOSE'])
    #     # position[index+1] = utils.ema_strategy(df['CLOSE'])
    #     position[index+1] = utils.swing_setup(df['HIGH'], df['LOW'], df['CLOSE'])
    #     index += 1    
    
    # Fit and predict
    prediction = pd.DataFrame(index=utils.futuresList)
    for (name, future), model in tqdm(settings['models'].items()):
        prediction.loc[future, name] = model.predict(data, future)
    sign = utils.sign(prediction)
    magnitude = utils.magnitude(prediction)
    
    # Futures strategy (Allocate position based on predictions)
    model = prediction.columns[0] # Arbitrarily pick first model in case of multiple
    position = basic_strategy(sign[model], magnitude[model]) 
    # position = long_only(sign[model], magnitude[model]) 
    # position = short_only(sign[model], magnitude[model]) 
        
    # Cash-futures strategy
    position = futures_only(position)

    # Update persistent data across runs
    settings['sign'].append(sign)
    settings['magnitude'].append(magnitude)

    # Update persistent data across runs
    settings['sign'].append(sign)
    settings['magnitude'].append(magnitude)

    # Yay!
    return position, settings


def mySettings():
    ''' Define your trading system settings here '''
    settings= {}
    settings['markets']  = utils.futuresAllList
    settings['beginInSample'] = '20190123'
    settings['endInSample'] = '20210331'
    settings['lookback']= 504
    settings['budget']= 10**6
    settings['slippage']= 0.05

    # Stuff to persist
    settings['models'] = LOADED_MODELS
    settings['sign'] = []
    settings['magnitude'] = []

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    quantiacsToolbox.runts(__file__)
