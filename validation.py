from models.numeric import (
    ArimaRaw, 
    ArimaLinear, 
    ArimaNoTrend, 
    ArimaLinearNoTrend,
)
from strategy import strategies_eval
import utils

# Must be imported last
from main import myTradingSystem, LOADED_MODELS

def mySettings():
    ''' Define your trading system settings here '''
    settings= {}
    settings['markets']  = utils.futuresAllList
    with open('windows.txt','r') as f:
        start, end = f.readline().split(", ")
        settings['beginInSample'] = start.replace('\'', '')
        settings['endInSample'] = end.replace('\'', '')
        # print(start, end)
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
    eval_res = strategies_eval.evaluate_by_sharpe(__file__, "19900101", "20201231", 27) # 2 years train 3 months val
    eval_res.to_csv('model_metrics/ihopethesharpeishigh.csv')
    print(eval_res)