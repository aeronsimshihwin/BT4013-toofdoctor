import utils
import math
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

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

INPUT = "future_preds_dict_1"
OUTPUT = "tuning_result_1"

def myTradingSystem(DATE, settings):
    ''' This system uses trend following techniques to allocate capital into the desired equities'''

    # Get saved X variables
    prediction = pd.DataFrame(index=utils.futuresList)

    with open(f"meta_model_predictions/{INPUT}.pkl", "rb") as f:
        # retrieve preds
        future_preds_dict = pickle.load(f)
        preds_dict_final = dict()

        for future in utils.futuresList:
            preds_dict_final[future] = future_preds_dict[future, settings['params']].copy()

    
    for future in tqdm(utils.futuresList):
        # read data
        try:
            future_pred = preds_dict_final[future].loc[datetime.strptime(str(DATE[-1]), '%Y%m%d')][0]
            prediction.loc[future, 'meta'] = future_pred
        except:
            print('ERROR: ', future, str(DATE[-1]))
            prediction.loc[future, 'meta'] = 0

    sign = utils.sign(prediction)
    magnitude = utils.magnitude(prediction)

    # position = basic_strategy(sign['meta'], magnitude['meta']) 
    position = fixed_threshold_strategy(sign['meta'], magnitude['meta'], settings['threshold'])

    # Cash-futures strategy
    position = futures_only(position)
    # position = futures_hold(position, settings['previous_position'])
    # position = cash_and_futures(position)

    # print(position)

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
    with open("utils/meta_tuning.txt", "r") as f2:
        params = f2.readline()

    with open("utils/meta_tuning_threshold.txt", "r") as f3:
        threshold = float(f3.readline())

    settings['params'] = params
    settings['threshold'] = threshold
    # settings['saved_predictions'] = preds_dict_final ## update this
    # print(preds_dict_final)
    settings['sign'] = []
    settings['magnitude'] = []
    settings['previous_position'] = np.array([0] * len(utils.futuresList) + [1,])

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox

    sharpe_results = pd.DataFrame(columns=['params', 'threshold', 'sharpe'])

    # parameter grid
    # xgbParams = [{'booster': ['gbtree'],
    #             'learning_rate': [0.01, 0.1, 0.3], # default 0.3
    #             'gamma': [0, 0.5, 1], # higher means more regularization
    #             'max_depth': [2, 4, 6, 8], # default 6
    # }]
    # final params
    xgbParams = [{'booster': ['gbtree'],
                'learning_rate': [0.01], # default 0.3
                'gamma': [1], # higher means more regularization
                'max_depth': [2]}]
    parameter_grid = list(ParameterGrid(xgbParams))

    # thresholds = [round(0.05 * x, 2) for x in range(0, 20)]
    # thresholds = [round(0.1 * x, 1) for x in range(0, 10)]
    thresholds = [0]

    for i in range(len(parameter_grid)):
        param_set = parameter_grid[i]
        params = f"lr{param_set['learning_rate']}_g{param_set['gamma']}_d{param_set['max_depth']}"

        with open ('utils/meta_tuning.txt', 'w') as file:
            file.write(params)

        for threshold in tqdm(thresholds):
            with open ('utils/meta_tuning_threshold.txt', 'w') as file:
                file.write(str(threshold))

            # retrieve sharpe
            results = quantiacsToolbox.runts(__file__, plotEquity=True)
            sharpe = results["stats"]["sharpe"]

            # save results
            print(params, threshold, sharpe)
            sharpe_results = sharpe_results.append({'params': params, 'threshold': threshold, 'sharpe': sharpe}, ignore_index=True)
    
    # sharpe_results.to_csv(f"meta_model_predictions/{OUTPUT}.csv", index=False)
