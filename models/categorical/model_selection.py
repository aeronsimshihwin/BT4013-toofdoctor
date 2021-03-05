import pandas as pd
import .utils

def save_model(txt_path, metric, model, X_train, y_train):
    '''
    Function that takes in model metrics txt path 'rf/perc/F_AD' and 
    selects the best model based on a metric.
    '''
    
    metrics_df = pd.read_csv(f'../../model_metrics/categorical/{txt_path}.txt')

    if metric[:8] == 'opp_cost':
        # select the row with lowest cost
        best_model = metrics_df.loc[metrics_df[metric] == min(metrics_df[metric])].reset_index(drop=True)
    else: # metric = 'accuracy
        # select the row with highest accuracy
        best_model = metrics_df.loc[metrics_df[metric] == max(metrics_df[metric])].reset_index(drop=True)
    
    params_dict = {}

    for col in metrics_df.columns():
        if col[:8] != 'opp_cost' and col[:8] != 'accuracy':
            params_dict[col] = best_model[col][0]
    
    # train and save model
    model = model(**params_dict)
    model.fit(X_train, y_train)

    with open(f'../../saved_models/categorical/{txt_path}.p', 'wb') as f:
        pickle.dump(model, f)

    return


# txt_path = 'rf/perc/F_AD'
# future = 'F_AD'

# # load data
# df = pd.read_csv(f"../../tickerData/{future}.txt", parse_dates = ["DATE"])
# df.columns = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL', 'OI', 'P', 'R', 'RINFO']
# df = df.set_index("DATE")
# df = df[(df.VOL != 0) & (df.CLOSE != 0)]
# df = df.dropna(axis=0)

# # load X and y
# X_df = utils.generate_X_df([df.CLOSE, df.VOL], ["perc", "perc"])
# y_df = utils.generate_y_cat(df.CLOSE)
