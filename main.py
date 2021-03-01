### Quantiacs Trend Following Trading System Example
# import necessary Packages below:
import numpy as np
import pandas as pd

from models import categorical, numeric, strategy
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
    
    # Collate additional information for models
    technical_indicators = utils.technical_indicators(OPEN, HIGH, LOW, CLOSE, VOL)
    economic_indicators = pd.DataFrame(
        data = np.array([
            USA_ADP, USA_EARN,\
            USA_HRS, USA_BOT, USA_BC, USA_BI, USA_CU, USA_CF, USA_CHJC, USA_CFNAI,\
            USA_CP, USA_CCR, USA_CPI, USA_CCPI, USA_CINF, USA_DFMI, USA_DUR,\
            USA_DURET, USA_EXPX, USA_EXVOL, USA_FRET, USA_FBI, USA_GBVL, USA_GPAY,\
            USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_CPIC, USA_CPICM,\
            USA_JBO, USA_LFPR, USA_LEI, USA_MPAY, USA_MP, USA_NAHB, USA_NLTTF,\
            USA_NFIB, USA_NFP, USA_NMPMI, USA_NPP, USA_EMPST, USA_PHS, USA_PFED,\
            USA_PP, USA_PPIC, USA_RSM, USA_RSY, USA_RSEA, USA_RFMI, USA_TVS, USA_UNR,\
            USA_WINV,
        ]).T,
        columns = (
            r'USA_ADP, USA_EARN,\
            USA_HRS, USA_BOT, USA_BC, USA_BI, USA_CU, USA_CF, USA_CHJC, USA_CFNAI,\
            USA_CP, USA_CCR, USA_CPI, USA_CCPI, USA_CINF, USA_DFMI, USA_DUR,\
            USA_DURET, USA_EXPX, USA_EXVOL, USA_FRET, USA_FBI, USA_GBVL, USA_GPAY,\
            USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_CPIC, USA_CPICM,\
            USA_JBO, USA_LFPR, USA_LEI, USA_MPAY, USA_MP, USA_NAHB, USA_NLTTF,\
            USA_NFIB, USA_NFP, USA_NMPMI, USA_NPP, USA_EMPST, USA_PHS, USA_PFED,\
            USA_PP, USA_PPIC, USA_RSM, USA_RSY, USA_RSEA, USA_RFMI, USA_TVS, USA_UNR,\
            USA_WINV'
        ).replace(' ', '').split(),
        index = pd.to_datetime(DATE, format='%Y%m%d'),
    )
    
    # Generate predictions, sign and magnitude for each model
    prediction = pd.DataFrame(index=utils.futuresList)
    sign = pd.DataFrame(index=utils.futuresList)
    magnitude = pd.DataFrame(index=utils.futuresList)

    for model_name, model_wrapper in categorical.models.items():
        probs = model_wrapper(
            OPEN, HIGH, LOW, CLOSE, VOL,
            technical_indicators = technical_indicators,
            economic_indicators = economic_indicators, 
        )
        prediction[model_name] = probs
        sign[model_name] = categorical.sign(probs)
        magnitude[model_name] = categorical.magnitude(probs)

    for model_name, model_wrapper in numeric.models.items():
        price_diff = model_wrapper(
            OPEN, HIGH, LOW, CLOSE, VOL,
            technical_indicators = technical_indicators,
            economic_indicators = economic_indicators, 
        )
        prediction[model_name] = price_diff
        sign[model_name] = numeric.sign(price_diff)
        magnitude[model_name] = numeric.magnitude(price_diff)

    # Determine trade positions for a set of strategies
    positions = pd.DataFrame(index=utils.futuresList)

    for model_name, model_wrapper in strategy.models.items():
        positions[model_name] = model_wrapper(
            OPEN, HIGH, LOW, CLOSE, VOL,
            technical_indicators = technical_indicators,
            economic_indicators = economic_indicators, 
            prediction = prediction,
            sign = sign,
            magnitude = magnitude,
        )

    # Consolidate positions from all strategies into a final position?
    # - Actually I think we are just hardcoding in the best strategy based on
    #   sharpe ratio right? Maybe someone can change this part haha
    aggregator = lambda x: x.mean(axis=1) # Stub
    position = aggregator(positions)

    # Cash-futures strategy
    mask = 1.0 # Stub
    position = mask * position

    # Normalization bc it doesn't hurt I guess
    weights = position/np.nansum(np.abs(position))

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