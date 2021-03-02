from dateutil.relativedelta import relativedelta
from datetime import date
from itertools import repeat
from typing import Callable, Mapping

import numpy as np
import pandas as pd

def _first_index_after(index: pd.Index, value):
    """Returns the min index i where i > value"""
    subindex = index.loc[index > value]
    if not subindex.empty:
        return subindex.min()
    else:
        raise ValueError

def _last_index_before(index, value):
    """Returns the max index i where i <= value"""
    subindex = index.loc[index <= value]
    if not subindex.empty:
        return subindex.max()
    else:
        raise ValueError

def simple_moving_average(x):
    return x.mean()

def exponential_moving_average(x):
    """Weighted by 1/(k+1), where k = distance from the most recent value"""
    return x @ np.arange(len(x), 0, -1)

def mean_square_error(y, y_pred):
    return np.mean(np.square(y - y_pred))

def walk_forward(
    model,
    data: pd.Series,
    metrics: Mapping[str, Callable] = {
        'MSE': mean_square_error,
    },
    aggregation: Mapping[str, Callable] = {
        'SMA': simple_moving_average, 
        'EMA': exponential_moving_average,
    },
    train_window: relativedelta = relativedelta(years=1),
    validation_window: relativedelta = relativedelta(months=3),
    start_index: date = None,
    max_windows: int = None,
    rolling = False,
):
    """Partitions the data into windows before evaluating model performance 
    within and across windows.

    Notes
    -----
    `model` argument: 
        Anything that has a fit method that takes in time series data, and a
        predict method that takes in the number of periods to predict.
    """
    if start_index is None:
        start_index = data.index.min()

    # Trace train window forwards in time
    train_start = _last_index_before(data.index, start_index)
    train_end = _last_index_before(data.index, train_start+train_window)

    # Trace validation window forwards in time
    validation_start = _first_index_after(data.index, train_end)
    validation_end = _last_index_before(data.index, validation_start+validation_window)

    # Generate all subsequent windows
    windows = [{
        'train_start': train_start,
        'train_end': train_end,
        'validation_start': validation_start,
        'validation_end': validation_end,
    }]

    for _ in repeat('I love making windows', times=max_windows): # Defaults to an infinite loop
        try:
            # From end of previous validation window, trace new train window backwards in time
            if rolling:
                train_start = _first_index_after(data.index, validation_end-train_window)
            train_end = validation_end
            train = (train_start >= data.index) & (data.index <= train_end)

            # From end of previous validation window, Trace new validation window forwards in time
            validation_start = _first_index_after(data.index, validation_end)
            validation_end = _last_index_before(data.index, validation_start+validation_window)        
            validation = (validation_start >= data.index) & (data.index <= validation_end)

            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'validation_start': validation_start,
                'validation_end': validation_end,
            })
        
        except ValueError:
            if max_windows is not None:
                print(f'warning: Insufficient data! Generated {len(windows)}/{max_windows} windows.')
            break
    
    windows = pd.DataFrame.from_records(windows)

    # Generate metrics
    results = pd.DataFrame(index=windows.index, columns=metrics.keys())
    for i in windows.index:
        train = (windows.loc[i, 'train_start'] >= data.index) & (data.index <= windows.loc[i, 'train_end'])
        validation = (windows.loc[i, 'validation_start'] >= data.index) & (data.index <= windows.loc[i, 'validation_end'])
        X = data.loc[train]
        y = data.loc[validation]
    
        fitted = model.fit(X)
        y_pred = fitted.predict(y)
        for name, func in metrics.items():
            results.loc[i, name] = func(y, y_pred)
        
    # Generate aggregated metrics
    aggregates = pd.DataFrame(index=aggregation.keys(), columns=metrics.keys())
    for name, func in aggregation.items():
        aggregates.loc[name] = results.apply(func, axis=0)
    
    # Yay!
    return windows, results, aggregates
