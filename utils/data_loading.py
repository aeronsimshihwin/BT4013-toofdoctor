from pathlib import Path
import pandas as pd
import utils

def load_futures_data(data_dir='tickerData'):
    """Simulates data passed to models in main.py"""
    import pandas as pd
    data = {}
    for future in utils.futuresList:
        df = pd.read_csv(f'{data_dir}/{future}.txt')
        df.columns = df.columns.str.strip()
        df.index = pd.to_datetime(df['DATE'], format='%Y%m%d')
        data[future] = df
    return data

def load_economic_indicators():
    """Simulates economic indicators in the standardized format"""
    results = dict()
    for p in Path('tickerData/').glob('USA_*.txt'):
        key = p.name.replace('USA_', '').replace('.txt', '')
        val = pd.read_csv(p, index_col=0)
        val.index = pd.to_datetime(val.index, format='%Y%m%d')
        results[key] = val
    return results

def load_predictions(path):
    """Loads predictions in similar form as standardized data"""
    root = Path(path)
    subdirs = [subdir for subdir in root.glob('*/') if subdir.is_dir()]
    results = dict()
    for future in utils.futuresList:
        y_preds = list()
        for subdir in subdirs:
            y_pred = pd.read_csv(subdir / f'{future}.csv', index_col=0)
            y_pred.columns = [subdir.name]
            y_pred.index = pd.to_datetime(y_pred.index)
            y_preds.append(y_pred)
        y_preds = pd.concat(y_preds, axis=1)
        y_preds = y_preds[y_preds.notna().any(axis=1)]
        results[future] = y_preds
    return results

