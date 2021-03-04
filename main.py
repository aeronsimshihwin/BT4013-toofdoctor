from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import strategy
import utils

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

    # Prepare standardized model input
    date_index = pd.to_datetime(DATE, format='%Y%m%d')
    data = dict()

    # Raw X data (This is here in case of disaster)
    # data.update({
    #     'OPEN': pd.DataFrame(OPEN, index=date_index, columns=utils.futuresAllList),
    #     'HIGH': pd.DataFrame(HIGH, index=date_index, columns=utils.futuresAllList),
    #     'LOW': pd.DataFrame(LOW, index=date_index, columns=utils.futuresAllList),
    #     'CLOSE': pd.DataFrame(CLOSE, index=date_index, columns=utils.futuresAllList),
    #     'VOL': pd.DataFrame(VOL, index=date_index, columns=utils.futuresAllList),
    # })

    for future in utils.futuresList:
        # Slice data by futures
        df = pd.DataFrame({
            'OPEN': OPEN[future],
            'HIGH': HIGH[future],
            'LOW': LOW[future],
            'CLOSE': CLOSE[future],
            'VOL': VOL[future],
        }, index=date_index)
        pass # Add technical_indicators as columns in each future dataframe
        pass # Add preprocessed features as columns in each future dataframe
        data[future] = df

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
    
    # Load saved models
    models = {}
    for model_dir in Path('saved_models').rglob('*/*'):
        if not model_dir.is_dir():
            continue
        *_, model_name = model_dir.parts
        for future in utils.futuresList:
            pickle_path = model_dir / f'{future}.p'
            try:
                with pickle_path.open('rb') as f:
                    models[model_name, future] = pickle.load(f)
            except:
                raise FileNotFoundError(f'No saved {model_name} for {future}!')

    # Fit and predict
    prediction = pd.DataFrame(index=utils.futuresList)
    for (name, future), model in models.items():
        prediction.loc[future, name] = model.predict(data[future])
    sign = utils.sign(prediction)
    magnitude = utils.magnitude(prediction)
    
    # Futures strategy (Allocate position based on predictions)
    position = data['prediction'].sample(axis='columns') # Stub: random sample

    # Cash-futures strategy
    cash_frac = 0.0
    weights = np.array([cash_frac, *position]) # Stub: ignore cash

    # Yay!
    return weights, settings


def mySettings():
    ''' Define your trading system settings here '''
    settings= {}
    settings['markets']  = utils.futuresAllList
    settings['beginInSample'] = '20190123'
    settings['endInSample'] = '20210331'
    settings['lookback']= 504
    settings['budget']= 10**6
    settings['slippage']= 0.05

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)