"""
What this does
--------------
1. Fits models on training data
2. Simulates a runts backtest over the entire dataset
3. Saves predictions made by each model
(4. Hopefully does all this more quickly using multiprocessing)
"""
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.numeric.arima import Arima
from models.numeric.model_validation import walk_forward
import utils

def load_and_preprocess_data():
    data = utils.load_futures_data()
    for future in tqdm(data):
        data[future] = utils.linearize(data[future], old_var='CLOSE', new_var='CLOSE_LINEAR')
        data[future] = utils.detrend(data[future], old_var='CLOSE_LINEAR', new_var='CLOSE_VELOCITY')
        data[future] = utils.detrend(data[future], old_var='CLOSE_VELOCITY', new_var='CLOSE_ACCELERATION')
    return data

def fit_model(candidate, data, future, force=False):
    # Load model
    try:
        if force:
            raise Exception
        with open(f'{candidate.SAVED_DIR}/{future}.p', 'rb') as f:
            model = pickle.load(f)
    
    # Or fit a new model if it doesn't exist
    except:
        model = Arima(y_var=candidate.y_var)
        model.fit(
            data, future,
            1, 1, 1,
            4, 2, 16,
            seasonal = False,
            suppress_warnings = True,
            error_action = 'ignore',
        )
        model.model.fit([0, 0, 0]) # Drop endog array before saving

    # Save the model
    for forecast in ('price', 'returns', 'percent'):    
        model.SAVED_DIR = model.SAVED_DIR.replace(model.forecast, forecast)
        model.forecast = forecast
        
        root = Path(model.SAVED_DIR)
        root.mkdir(parents=True, exist_ok=True)
        p = root / f'{future}.p'
        with p.open('wb') as f:
            pickle.dump(model, f)

def load_models(candidates):
    records = []
    for candidate in tqdm(candidates):
        root = candidate.SAVED_DIR.replace(f'/{candidate.forecast}', '')
        for forecast in ('price', 'returns', 'percent'):    
            for future in data:
                with open(f'{root}/{forecast}/{future}.p', 'rb') as f:
                    records.append({
                        'model': root.split('/')[-1],
                        'forecast': forecast,
                        'future': future,
                        'arima': pickle.load(f),
                    })
    models = pd.DataFrame.from_records(records)
    models = models.set_index(['model', 'forecast', 'future'])
    models['name'] = models['arima'].apply(lambda x: str(x.model))
    return models

def predict_future(model, data, future, force=False):
    root = Path('model_predictions/numeric/arima')
    root.mkdir(parents=True, exist_ok=True)
    p = root / f'{future}.csv'
    try:
        if force:
            raise Exception
        pd.read_csv(p, index_col=0)
    except:
        windows, y_preds = walk_forward(
            model = model,
            data = data[future]['CLOSE_LINEAR'],
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
    candidates = [
        Arima('CLOSE_LINEAR'),
        Arima('CLOSE_VELOCITY'),
        Arima('CLOSE_ACCELERATION'),
    ]
    
    print('Loading and preprocessing data...')
    data = load_and_preprocess_data()

    print('Fitting models')
    with Pool(processes=num_cores) as pool:
        args = [
            (candidate, data, future, flags.force_fit) 
            for candidate in candidates 
            for future in data
        ]
        pool.starmap(fit_model, tqdm(args, position=0))

    print('Loading models...')
    models = load_models(candidates)

    print('Model predictions...')
    with Pool(processes=num_cores) as pool:
        arima = models['arima']
        args = [
            (arima['CLOSE_LINEAR', 'price', future].model, data, future, flags.force_predict) 
            for future in data
        ]
        pool.starmap(predict_future, tqdm(args, position=0))