import utils
import pandas as pd
from sklearn.metrics import accuracy_score
from itertools import repeat
from typing import Callable, Mapping
from datetime import date, datetime
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

def walk_forward(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cost_weight: pd.Series,
    aggregation: Mapping[str, Callable] = {
        'SMA': utils.simple_moving_average, 
        # 'EMA': utils.exponential_moving_average,
    },
    train_window: relativedelta = relativedelta(years=2),
    validation_window: relativedelta = relativedelta(months=3),
    start_index: date = None,
    max_windows: int = 10**3,
    rolling = False,
):
    X = X.dropna(axis=0)
    y = y.dropna(axis=0)
    cost_weight = cost_weight.dropna(axis=0)
    
    # get intersection of all dataframes
    common_index = X.index.intersection(y.index).intersection(cost_weight.index)
    X = X[X.index.isin(common_index)]
    y = y[y.index.isin(common_index)]
    cost_weight = cost_weight[cost_weight.index.isin(common_index)]
    
    if start_index is None:
        start_index = X.index.min()
    else:
        start_index = max(start_index, X.index.min())

    # Trace train window forwards in time
    train_start = _first_quarter_after(start_index)
    train_end = train_start + train_window

    # Trace validation window forwards in time
    validation_start = train_end
    validation_end = validation_start + validation_window

    # Generate all subsequent windows
    windows = [{
        'train_start': train_start,
        'train_end': train_end,
        'validation_start': validation_start,
        'validation_end': validation_end,
    }]

    for _ in repeat('I love making windows', max_windows): # Defaults to an infinite loop
        # From end of previous validation window, trace new train window backwards in time
        if rolling:
            train_start = _first_quarter_after(validation_end - train_window)
        train_end = validation_end

        # From end of previous validation window, Trace new validation window forwards in time
        validation_start = _first_quarter_after(validation_end)
        validation_end = validation_start + validation_window    

        if validation_end > y.index[-1]:
            break

        windows.append({
            'train_start': train_start,
            'train_end': train_end,
            'validation_start': validation_start,
            'validation_end': validation_end,
        })

    windows = pd.DataFrame.from_records(windows)

    # Generate metrics
    metrics = ["accuracy", "opp_cost"]
    results = pd.DataFrame(index=windows.index, columns=metrics)
    for i in windows.index:
        train_mask = (X.index >= windows.loc[i, 'train_start']) & (X.index < windows.loc[i, 'train_end'])
        val_mask = (X.index >= windows.loc[i, 'validation_start']) & (X.index < windows.loc[i, 'validation_end'])
        X_train, X_val = X.loc[train_mask].to_numpy(), X.loc[val_mask].to_numpy()
        y_train, y_val = y.loc[train_mask].to_numpy(), y.loc[val_mask].to_numpy()
        cost_val = cost_weight.loc[val_mask].to_numpy()

        fitted = model.fit(X_train, y_train)
        y_pred = fitted.predict(X_val) 
        results.loc[i, "accuracy"] = accuracy_score(pd.Series(y_val), pd.Series(y_pred))
        results.loc[i, "opp_cost"] = utils.opportunity_cost(pd.Series(y_val), pd.Series(y_pred), pd.Series(cost_val))

    # Combine windows with metric results
    win_results = pd.concat([windows, results], axis=1)

    agg_results = pd.DataFrame(index=aggregation.keys(), columns=metrics)
    for name, func in aggregation.items():
        agg_results.loc[name] = results.apply(func, axis=0)

    # Yay!
    return win_results, agg_results
