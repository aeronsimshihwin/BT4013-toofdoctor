from pathlib import Path
import pickle

def fit_model(args):
    candidate, data, future, force = args
    root = Path(candidate.SAVED_DIR)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{future}.p'
    try:
        if force:
            raise Exception
        with path.open('rb') as f:
            model = pickle.load(f)
    except:
        model = candidate(future=future)
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

    num_cores = cpu_count()

    print('Loading and preprocessing data...')
    data = utils.load_futures_data()

    print('Fitting models')
    np.seterr(all='warn') # Allow numpy warnings to be handled
    with Pool(processes=num_cores) as pool:
        args = [
            (ArimaEnsemble, data, future, flags.force_fit)  
            for future in data
        ]
        progress = pool.imap_unordered(fit_model, args) # Iterator of results
        for _ in tqdm(progress, total=len(args), position=0):
            pass
