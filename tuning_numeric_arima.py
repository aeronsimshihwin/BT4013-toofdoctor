"""
What this does
--------------
1. Fits models on training data
2. Simulates a runts backtest over the entire dataset
3. Saves predictions made by each model
4. Does all this more quickly using multiprocessing . ^ .
"""
from datetime import datetime
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.numeric.arima import Arima
from models.numeric.model_validation import walk_forward
import utils

# def load_and_preprocess_data():
#     data = utils.load_futures_data()
#     for future in tqdm(data):
#         data[future] = utils.linearize(data[future], old_var='CLOSE', new_var='CLOSE_LINEAR')
#         data[future] = utils.detrend(data[future], old_var='CLOSE_LINEAR', new_var='CLOSE_VELOCITY')
#         data[future] = utils.detrend(data[future], old_var='CLOSE_VELOCITY', new_var='CLOSE_ACCELERATION')
#     return data

# def load_and_preprocess_data(future):
#     data = prepare_data(future)
#     data = data.join(indicators)
#     return data

def fit_model(args):
    # Lousy hack for multiprocessing.Pool.imap_unordered
    model, data, future, force = args

    root = Path(f'{model.SAVED_DIR}/{model.y}')
    root.mkdir(parents=True, exist_ok=True)
    p = root / f'{future}.p'

    # Load model
    try:
        if force:
            raise Exception
        with p.open('rb') as f:
            model = pickle.load(f)
    
    # Or fit a new model if it doesn't exist
    except:
        data = {
            future: df[df.index < datetime(2021, 1, 1)] # Exclude test data
            for future, df in data.items()
        }
        try:
            model.fit(
                data, future,
                start_p=1, start_d=1, start_q=1,
                max_p=4, max_d=2, max_q=16,
                # seasonal = False,
                suppress_warnings = True,
                error_action = 'ignore',
                random_state = 0,
            )
        except:
            # Fall back to SARIMA
            model.X = None
            model.fit(
                data, future,
                start_p=1, start_d=1, start_q=1,
                max_p=4, max_d=2, max_q=16,
                # seasonal = False,
                suppress_warnings = True,
                error_action = 'ignore',
            )

        # Try shrink endog array before saving
        for n in (2, 5, 10, 15, 20):
            try:
                model.model.fit(n*[0])
                break
            except:
                continue

        # Save the model    
        with p.open('wb') as f:
            pickle.dump(model, f)

# def load_models(candidates):
#     records = []
#     for candidate in tqdm(candidates):
#         root = candidate.SAVED_DIR.replace(f'/{candidate.forecast}', '')
#         for forecast in ('price', 'returns', 'percent'):    
#             for future in data:
#                 with open(f'{root}/{forecast}/{future}.p', 'rb') as f:
#                     records.append({
#                         'model': root.split('/')[-1],
#                         'forecast': forecast,
#                         'future': future,
#                         'arima': pickle.load(f),
#                     })
#     models = pd.DataFrame.from_records(records)
#     models = models.set_index(['model', 'forecast', 'future'])
#     models['name'] = models['arima'].apply(lambda x: str(x.model))
#     models = models.sort_index()
#     return models

def predict_future(args):
    # Lousy hack for multiprocessing.Pool.imap_unordered
    model, data, future, force = args
    
    root = Path(f'model_predictions/numeric/arima/{model.y}')
    root.mkdir(parents=True, exist_ok=True)
    p = root / f'{future}.csv'
    try:
        if force:
            raise Exception
        pd.read_csv(p, index_col=0)
    except:
        windows, y_preds = walk_forward(
            model = model,
            data = data,
            future = future,
            progress_bar = True,
        )
        y_preds.to_csv(p)

if __name__ == '__main__':
    import argparse
    from multiprocessing import Pool, cpu_count
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force-fit', action='store_true', help='Overwrites existing models if they exist')
    parser.add_argument('--force-predict', action='store_true', help='Overwrites existing predictions if they exist')
    flags = parser.parse_args()

    if flags.force_fit and not flags.force_predict:
        warnings.warn('Force fitting models without generating new predictions!')

    num_cores = cpu_count()
    y_vars = ['CLOSE_LINEAR', 'CLOSE_VELOCITY', 'CLOSE_ACCELERATION']
    
    print('Loading and preprocessing data...')
    # data = load_and_preprocess_data()
    with Pool(processes=num_cores) as pool:
        results = pool.imap(utils.prepare_data, utils.futuresList) # Iterator of results
        results = tqdm(results, total=len(utils.futuresList), position=0) # Status bar
        data = dict(zip(utils.futuresList, results))
    data = {future: df.dropna() for future, df in data.items()}

    print('Fitting models')
    np.seterr(all='warn') # Allow numpy warnings to be handled
    with Pool(processes=num_cores) as pool:
        args = [
            (
                Arima(y=y), 
                {future: data[future]}, 
                future, 
                flags.force_fit,
            )
            for y in y_vars
            for future in data
        ]
        progress = pool.imap_unordered(fit_model, args) # Iterator of results
        for _ in tqdm(progress, total=len(y_vars)*len(data), position=0):
            pass
    
    print('Loading models...')
    # models = load_models(y_vars)
    records = []
    for future in tqdm(utils.futuresList):
        for y in y_vars:
            with open(f'{Arima.SAVED_DIR}/{y}/{future}.p', 'rb') as f:
                records.append({
                    'future': future,
                    'y': y,
                    'arima': pickle.load(f),
                })
    models = pd.DataFrame.from_records(records)
    models['name'] = models['arima'].apply(lambda x: str(x.model))
    models = models.set_index(['future', 'y'])
    models = models.sort_index()

    print('Model predictions...')
    np.seterr(all='warn') # Allow numpy warnings to be handled
    with Pool(processes=num_cores) as pool:
        arima = models['arima']
        args = [
            (arima[future, y], data, future, flags.force_predict)
            for y in y_vars
            for future in data
        ]
        progress = pool.imap_unordered(predict_future, args) # Iterator of results
        for _ in tqdm(progress, total=len(args), position=0):
            pass
    