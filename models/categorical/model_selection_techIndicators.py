import utils
import pandas as pd
import numpy as np
import pickle
import math
from itertools import repeat
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

def _last_quarter_before(start_index):
    month = (start_index.month - 1)//3*3 + 1
    return datetime(start_index.year, month, 1)
    
def _first_quarter_after(start_index):
    year = start_index.year
    month = start_index.month
    if datetime(year, (month-1)//3*3 + 1, 1) == start_index:
        return start_index
    else:
        start_index = start_index + relativedelta(months=3)
        month = (start_index.month - 1)//3*3 + 1
        return datetime(start_index.year, month, 1)

def save_meta_predictions_techIndicators(
    path, metric, future, X_vars, y_var, model_name, ext_path="csv", \
    start_index:datetime = datetime(2016,10,1), end_index:datetime = datetime(2021,1,1), \
    train_window: relativedelta = relativedelta(years=2),
    validation_window: relativedelta = relativedelta(months=3)):
    
    # generate data
    df = utils.prepare_data(future)
    X, y = utils.generate_X_y(df, X_vars=X_vars, y_var=y_var)

    ## GENERATE WINDOWS ##
    # trace train window forwards in time
    train_start = _first_quarter_after(start_index)
    train_end = train_start + train_window

    # trace validation window forwards in time
    validation_start = train_end
    validation_end = validation_start + validation_window

    # generate all subsequent windows
    windows = [{
        'train_start': train_start,
        'train_end': train_end,
        'validation_start': validation_start,
        'validation_end': validation_end,
    }]
    for _ in repeat('I love making windows', 20): # Defaults to an infinite loop
        # from end of previous validation window, trace new train window backwards in time
        train_start = _first_quarter_after(validation_end - train_window)
        train_end = validation_end

        # from end of previous validation window, Trace new validation window forwards in time
        validation_start = _first_quarter_after(validation_end)
        validation_end = validation_start + validation_window    

        if validation_end > end_index:
            break
        
        windows.append({
            'train_start': train_start,
            'train_end': train_end,
            'validation_start': validation_start,
            'validation_end': validation_end,
        })

    windows = pd.DataFrame.from_records(windows)

    # RETRIEVE OPTIMAL PARAMETERS ##
    # convert metrics txt file to pd.DataFrame
    metrics_df = pd.read_csv(f'model_metrics/categorical/{path}{future}.{ext_path}')

    # set objective based on input metric
    if metric[:8] == 'opp_cost':
        # select the row with lowest cost
        best_metric = metrics_df.loc[metrics_df[metric] == min(metrics_df[metric])].reset_index(drop=True)
    else: # metric = 'accuracy'
        # select the row with highest accuracy
        best_metric = metrics_df.loc[metrics_df[metric] == max(metrics_df[metric])].reset_index(drop=True)
    
    # retrieve optimal parameters corresponding to best metric
    params_dict = {}
    for col in metrics_df.columns:
        if col[:8] != 'opp_cost' and col[:8] != 'accuracy' and \
            not (isinstance(best_metric[col][0], float) and math.isnan(best_metric[col][0])):
            params_dict[col] = best_metric[col][0]

    predictions = pd.Series()
    for i in windows.index:
        train_mask = (X.index >= windows.loc[i, 'train_start']) & (X.index < windows.loc[i, 'train_end'])
        val_mask = (X.index >= windows.loc[i, 'validation_start']) & (X.index < windows.loc[i, 'validation_end'])
        X_train, X_val = X.loc[train_mask].to_numpy(), X.loc[val_mask].to_numpy()
        y_train, y_val = y.loc[train_mask].to_numpy(), y.loc[val_mask].to_numpy()
        
        if model_name == "fourCandleHammer":
            df_with_strategy = utils.fourCandleHammer(X, params_dict['N'], params_dict['highFactor'], params_dict['lowFactor'], future, params_dict['macro_analysis'])
        elif model_name == "ema_strategy":
            df_with_strategy = utils.ema_strategy(X, params_dict['shortTermDays'], params_dict['longTermDays'], params_dict['NDays'], future, params_dict['macro_analysis'])
        elif model_name == "swing_setup":
            df_with_strategy = utils.swing_setup(X, params_dict['shortTermDays'], params_dict['longTermDays'], params_dict['NDays'], future, params_dict['macro_analysis'])
        # fitted = model.fit(X_train, y_train)
        # y_pred = fitted.predict_proba(X_val)[:,1]
        # y_pred_series = pd.Series(y_pred, index=X.loc[val_mask].index)
        long_short = df_with_strategy['LONG_SHORT']
        if -100 in long_short: # In first window, without sufficient "N" data points, strategy will not output long/short/do nothing.
            unique_val, counts = np.unique(long_short, return_counts=True)
            unique_val_count = dict(zip(unique_val, counts))
            numOfInvalids = unique_val_count[-100]
            long_short = long_short[numOfInvalids:]
            y_val = y_val[numOfInvalids:]
        y_pred_series = pd.Series(long_short, index=X.loc[val_mask].index)
        predictions = predictions.append(y_pred_series)

    predictions_df = pd.DataFrame(predictions, columns=[model_name])
    predictions_df.to_csv(f"meta_model_predictions/categorical/{path}{future}.csv")

def save_model_techIndicators(path, metric, model_wrapper, future, X_vars, y_var, ext_path="csv",\
    train_start=date(2019,1,1), train_end=date(2021,1,1)):
    '''
    Function that takes in model metrics txt path, 
    selects the best model parameters based on a metric,
    trains the model then saves it in a .p file.

    Inputs:
    txt_path (str): e.g. 'rf/perc/F_AD' in format 'model_name/x_var/future'
    metric (str): one of 'accuracy_SMA', 'accuracy_EMA', 'opp_cost_SMA', 'opp_cost_EMA'
    X_train (numpy array)
    y_train (numpy array)

    Output:
    None
    '''
    # prepare train and test data
    df = utils.prepare_data(future)
    X_df, y_df = utils.generate_X_y(df, X_vars=X_vars, y_var=y_var)
    X_train = X_df[train_start:train_end]
    y_train = y_df[train_start:train_end]

    # convert metrics txt file to pd.DataFrame
    metrics_df = pd.read_csv(f'model_metrics/categorical/{path}/{future}.{ext_path}')

    # set objective based on input metric
    if metric[:8] == 'opp_cost':
        # select the row with lowest cost
        best_metric = metrics_df.loc[metrics_df[metric] == min(metrics_df[metric])].reset_index(drop=True)
    else: # metric = 'accuracy'
        # select the row with highest accuracy
        best_metric = metrics_df.loc[metrics_df[metric] == max(metrics_df[metric])].reset_index(drop=True)
    
    # retrieve optimal parameters corresponding to best metric
    params_dict = {}
    for col in metrics_df.columns:
        if col[:8] != 'opp_cost' and col[:8] != 'accuracy' and \
            not (isinstance(best_metric[col][0], float) and math.isnan(best_metric[col][0])):
            params_dict[col] = best_metric[col][0]
    
    # train model using optimal parameters
    try:
        # model = model_fn(**params_dict)
        # fitted = model.fit(X_train, y_train)
        save = model_wrapper(model_params=params_dict, X=X_vars, y=y_var)
    except:
        print(params_dict)
        return (X_train, y_train)
    
    with open(f'saved_models/categorical/{path}/{future}.p', 'wb') as f:
        pickle.dump(save, f)

    return save
