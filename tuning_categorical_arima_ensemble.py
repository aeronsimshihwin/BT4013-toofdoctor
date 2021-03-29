from datetime import datetime
from pathlib import Path
import pickle

def fit_model(args):
    model, data, future, force = args
    root = Path(model.SAVED_DIR)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{future}.p'
    try:
        if force:
            raise Exception
        with path.open('rb') as f:
            model = pickle.load(f)
    except:
        data = {
            future: df[df.index < datetime(2021, 1, 1)] # Exclude test data
            for future, df in data.items()
        }
        model.fit(data, future)
        with path.open('wb') as f:
            pickle.dump(model, f)

if __name__ == '__main__':
    import argparse
    from multiprocessing import Pool, cpu_count
    import warnings
    
    import numpy as np
    from tqdm import tqdm
    
    from models.categorical.arima_ensemble import ArimaEnsemble
    import utils
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force-fit', action='store_true', help='Overwrites existing models if they exist')
    parser.add_argument('--force-predict', action='store_true', help='Overwrites existing predictions if they exist')
    flags = parser.parse_args()

    if flags.force_fit and not flags.force_predict:
        warnings.warn('Force fitting models without generating new predictions!')

    def fit_args(candidates):
        for name, (candidate, param_grid) in candidates.items():
            for future in utils.futuresList:
                model = ArimaEnsemble(
                    future = future,
                    xgb_features = [
                        *['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL'],
                        *utils.generate_X_vars(future, tech_indicators=[], macro_indicators=True),
                    ],
                    model = GridSearchCV(
                        estimator = candidate,
                        param_grid = param_grid,
                        cv = TimeSeriesSplit(5),
                        return_train_score = True,
                    ),
                )
                model.SAVED_DIR = f'{model.SAVED_DIR}/{name}'
                yield (model, {future: data[future]}, future, flags.force_fit)
    
    num_cores = cpu_count()

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
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        from sklearn.neural_network import MLPClassifier
        from sklearn.tree import DecisionTreeClassifier
        from xgboost import XGBClassifier

        candidates = {
            'mlp': (
                MLPClassifier(solver='adam', max_iter=10000, early_stopping=True, random_state=0),
                {
                    'activation': ['relu', 'logistic'],
                    'hidden_layer_sizes': [(8,), (8,4)],
                    'alpha': np.logspace(-4, 0, 5),
                },
            ),
            'tree': (
                DecisionTreeClassifier(random_state=0),
                {
                    'max_depth': [3, 4, 5, 6], 
                    'ccp_alpha': np.logspace(-4, 0, 5),
                },
            ),
            'xgb': (
                XGBClassifier(eval_metric='mlogloss', random_state=0),
                {
                    'max_depth': [3, 4, 5, 6], 
                    'learning_rate': np.logspace(-4, 0, 5),
                },
            ),
        }

        progress = pool.imap_unordered(fit_model, fit_args(candidates)) # Iterator of results
        num_candidates = len(candidates)
        num_futures = len(utils.futuresList)
        for _ in tqdm(progress, total=num_candidates*num_futures, position=0):
            pass
