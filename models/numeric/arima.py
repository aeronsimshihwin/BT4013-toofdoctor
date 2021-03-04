import numpy as np
import pmdarima as pm

SAVED_DIR = 'saved_models/numeric/arima'

class ArimaWrapper:
    def __init__(self, model=None, y=None, X=None):
        self.model = model
        self.get_y = y # Extracts y from data[future]
        self.get_X = X # Extracts X from data[future]
    
    def fit(self, data, future):
        X = None if self.get_X is None else self.get_X(data[future])
        y = None if self.get_y is None else self.get_y(data[future])
        if self.model is None:
            self.model = pm.auto_arima(y, error_action='ignore')
        else:
            self.model.update(y.iloc[-1])
    
    def predict(self, data, future):
        y_diff = np.nan # Default return value
        if future in data:
            self.fit(y)
            y_pred = self.model.predict(n_periods=1)[0]
            y_diff = y_pred - y.iloc[-1]
        return y_diff

# class ArimaWrapper:
#     def __init__(self, models={}, y=None, X=None):
#         self.models = models
#         self.get_y = y # Extracts y from data[future]
#         self.get_X = X # Extracts X from data[future]
        
#     def save(self, save_dir):
#         path = Path(save_dir)
#         path.mkdir(parents=True, exist_ok=True)
#         for future in utils.futuresList:
#             filepath = path / f'{future}.p'
#             with filepath.open('wb') as f:
#                 pickle.dump(self.models[future], f)
            
#     def load(self, save_dir):
#         self.models = dict()
#         for future in utils.futuresList:
#             with open(f'{save_dir}/{future}.p', 'rb') as f:
#                 self.models[future] = pickle.load(f)
    
#     def fit_predict(self, data, trace=None):
#         y_diff = []        
#         for future in tqdm(utils.futuresList):
#             if future in data:
#                 X = None if self.get_X is None else self.get_X(data[future])
#                 y = None if self.get_y is None else self.get_y(data[future])
#                 y_last = y.iloc[-1]
#             else:
#                 y_diff.append(np.nan)
#                 continue
            
#             if future in self.models:
#                 self.models[future].update(y_last)
#             else:
#                 self.models[future] = pm.auto_arima(y, trace=trace)
            
#             y_pred = self.models[future].predict(n_periods=1)[0]
#             y_diff.append(y_pred - y_last)                
                
#         return pd.Series(y_diff, index=utils.futuresList)
