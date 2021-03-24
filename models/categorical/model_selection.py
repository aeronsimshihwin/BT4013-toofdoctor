import utils
import pandas as pd
import numpy as np
import pickle
import math
from datetime import date

def save_model(path, metric, model_fn, model_wrapper, future, X_vars, y_var, ext_path="csv",\
    train_start=date(2019,1,1), train_end=date(2021,1,1)):
    '''
    Function that takes in model metrics txt path, 
    selects the best model parameters based on a metric,
    trains the model then saves it in a .p file.

    Inputs:
    txt_path (str): e.g. 'rf/perc/F_AD' in format 'model_name/x_var/future'
    metric (str): one of 'accuracy_SMA', 'accuracy_EMA', 'opp_cost_SMA', 'opp_cost_EMA'
    model (sklearn model specification): e.g. RandomForestClassifier, LogisticRegression
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
        model = model_fn(**params_dict)
        fitted = model.fit(X_train, y_train)
        save = model_wrapper(model=fitted, X=X_vars, y=y_var)
    except:
        print(params_dict)
        return (X_train, y_train)
    
    with open(f'saved_models/categorical/{path}/{future}.p', 'wb') as f:
        pickle.dump(save, f)

    return save
