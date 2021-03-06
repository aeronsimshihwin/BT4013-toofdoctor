import utils
import pandas as pd
import numpy as np
import pickle

def save_model(txt_path, metric, model, X_train, y_train, ext_path="txt"):
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
    # convert metrics txt file to pd.DataFrame
    metrics_df = pd.read_csv(f'model_metrics/categorical/{txt_path}.{ext_path}')

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
        if col[:8] != 'opp_cost' and col[:8] != 'accuracy':
            params_dict[col] = best_metric[col][0]
    
    # train model using optimal parameters
    model = model(**params_dict)
    fitted = model.fit(X_train, y_train)

    with open(f'saved_models/categorical/{txt_path}.p', 'wb') as f:
        pickle.dump(model, f)

    return fitted
