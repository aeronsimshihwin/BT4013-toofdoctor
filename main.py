import numpy as np
import pandas as pd

from models import categorical, numeric
import strategy
import utils

def prepare_data(DATE, OPEN, HIGH, LOW, CLOSE, VOL, USA_ADP, USA_EARN,\
    USA_HRS, USA_BOT, USA_BC, USA_BI, USA_CU, USA_CF, USA_CHJC, USA_CFNAI,\
    USA_CP, USA_CCR, USA_CPI, USA_CCPI, USA_CINF, USA_DFMI, USA_DUR,\
    USA_DURET, USA_EXPX, USA_EXVOL, USA_FRET, USA_FBI, USA_GBVL, USA_GPAY,\
    USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_CPIC, USA_CPICM,\
    USA_JBO, USA_LFPR, USA_LEI, USA_MPAY, USA_MP, USA_NAHB, USA_NLTTF,\
    USA_NFIB, USA_NFP, USA_NMPMI, USA_NPP, USA_EMPST, USA_PHS, USA_PFED,\
    USA_PP, USA_PPIC, USA_RSM, USA_RSY, USA_RSEA, USA_RFMI, USA_TVS, USA_UNR,\
    USA_WINV, exposure, equity, settings):
    """Returns a dict of dataframes that serves as the standardized data input for all models.
    
    Data specification
    ------------------
    Each DataFrame in the dict should correspond to a future or an economic indicator.
    - ? x futures
    - ? x economic indicators
    
    DataFrame specification
    -----------------------
    Each column corresponds to some time series associated with the future / economic indicator
    - Index: datetime
    - Columns: Price data type: OPEN / CLOSE/ technical indicator name / preprocessed feature name / etc.
    """
    date_index = pd.to_datetime(DATE, format='%Y%m%d')
    data = dict()
    
    # Raw X data
    # data.update({
    #     'OPEN': pd.DataFrame(OPEN, index=date_index, columns=utils.futuresAllList),
    #     'HIGH': pd.DataFrame(HIGH, index=date_index, columns=utils.futuresAllList),
    #     'LOW': pd.DataFrame(LOW, index=date_index, columns=utils.futuresAllList),
    #     'CLOSE': pd.DataFrame(CLOSE, index=date_index, columns=utils.futuresAllList),
    #     'VOL': pd.DataFrame(VOL, index=date_index, columns=utils.futuresAllList),
    # })
    
    # Pre-processed X data
    for future in utils.futuresList:
        # Slice data by futures
        df = pd.DataFrame({
            'OPEN': OPEN[future],
            'HIGH': HIGH[future],
            'LOW': LOW[future],
            'CLOSE': CLOSE[future],
            'VOL': VOL[future],
        }, index=date_index)
        
        # Technical_indicators
        # df['technical_indicator_1'] = df['CLOSE'] - df['OPEN']
        # df['technical_indicator_2'] = np.nan
        # df['technical_indicator_3'] = np.nan

        # Collate additional information for models
        # df['categorical_preprocessing_1'] = np.nan
        # df['arima_preprocessing_1'] = np.nan

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
    
    return data
    

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

    for future in utils.futuresList:
        # Slice data by futures
        df = pd.DataFrame({
            'OPEN': OPEN[future],
            'HIGH': HIGH[future],
            'LOW': LOW[future],
            'CLOSE': CLOSE[future],
            'VOL': VOL[future],
        }, index=date_index)
        pass # Technical_indicators
        pass # Preprocessed features
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
    models = dict()
    for (name, Model, args, kwargs, save_file) in categorical.models + numeric.models:
        models[name] = Model(*args, **kwargs)
        models[name].load(save_file)

    # Fit and predict
    prediction = pd.DataFrame(index=utils.futuresList)
    sign = pd.DataFrame(index=utils.futuresList)
    magnitude = pd.DataFrame(index=utils.futuresList)
    
    for name, model in models.items():
        # Models choose which predictors to use in implementation
        # Predictions in pd.Series with index = utils.futuresList
        preds = model.fit_predict(data)

        prediction[name] = preds
        sign[name] = utils.sign(preds)
        magnitude[name] = utils.magnitude(preds)

    # Allow strategy stage to access all prediction in addition to raw data
    data['prediction'] = prediction
    data['sign'] = sign
    data['magnitude'] = magnitude

    # Futures strategy
    position = data['prediction'].sample(axis='columns') # Stub: random sample

    # Cash-futures strategy
    cash_frac = 0.0
    weights = np.array([cash_frac, *position]) # Stub: ignore cash

    # Yay!
    print('Another day passes...')
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