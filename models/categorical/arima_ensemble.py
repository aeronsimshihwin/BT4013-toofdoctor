"""

"""
import pickle
from typing import List
import warnings

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import utils

class ArimaEnsemble:
    """Feeds raw data and arima forecasts to an XGBoost classifier to predict positions"""
    SAVED_DIR = 'saved_models/categorical/arima_ensemble'
    ARIMA_DIR = 'saved_models/numeric/arima'

    def __init__(
        self, 
        future: str = None,
        model: XGBClassifier = None,
        xgb_features: List[str] = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL'], 
        arima_features: List[str] = ['CLOSE_LINEAR', 'CLOSE_VELOCITY', 'CLOSE_ACCELERATION'],
    ):
        """Load pre-trained ARIMA models"""
        self.future = None
        self.model = model
        self.xgb_features = xgb_features
        self.arima_features  = arima_features
        self.arima = dict()
        self._load_arima(future)
        
    def _load_arima(self, future):
        """Loads appropriate pre-trained ARIMA models for a particular future"""
        if future != self.future:
            self.future = future
            self.arima = dict()
            for y_var in self.arima_features:
                with open(f'{self.ARIMA_DIR}/{y_var}/{future}.p', 'rb') as f:
                    self.arima[y_var] = pickle.load(f)

    def fit(self, data, future, **kwargs):
        """Fits an XGBoostClassifier from cached ARIMA predictions"""
        self._load_arima(future)

        # Extract raw features for XGBoost
        X1 = data[future]
        X1 = X1[self.xgb_features]

        # Load cached predictions for ARIMA submodels
        preds_dir = self.ARIMA_DIR.replace('saved_models', 'model_predictions')
        arima_pred = utils.load_predictions(preds_dir)
        X2 = arima_pred[future]
        X2 = X2[self.arima_features]
        X2 = X2.shift(-1) # Forecasts made AT each date (instead of predictions FOR each date)

        # Extract position labels from data
        labels = data[future]['CLOSE']
        labels = labels.diff().shift(-1) # Optimal position for the NEXT day (instead of CURRENT day)
        labels = np.sign(labels.dropna())
        labels.name = 'label'

        # Collate features and align indices
        features = pd.concat([X1, X2, labels], axis=1)
        features = features.dropna()
        y = 'label'
        X = features.columns.drop(y)
        
        # Default model (Consider wrapping in GridSearchCV to tune hyperparams if performance is poor)
        if self.model is None:
            self.model = XGBClassifier(eval_metric='mlogloss')
            # self.model = GridSearchCV(
            #     estimator = XGBClassifier(eval_metric='mlogloss'),
            #     param_grid = {
            #         # params to tune here
            #     },
            #     scoring = None, # Can put in our opp cost function somehow?
            #     cv = 10, # 10-fold cross validation
            # )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # .fit throws some deprecation warnings for default params
            self.model.fit(
                features[X], features[y],
                # eval_set = [(features[X], features[y])],
                # eval_metric = 'mlogloss',
                # verbose = False,
            )
        
        return self

    def predict(self, data, future, threshold=0.3):
        """Collates predictions from arima sub-models and passes them to a XGBoost meta-classifier"""
        self._load_arima(future)
        
        price_data = data[future][self.xgb_features].tail(1)
        arima_pred = pd.DataFrame({
            feature: [self.arima[feature].predict(data, future)]
            for feature in self.arima_features
        })

        features = pd.concat([
            price_data.reset_index(drop=True), 
            arima_pred.reset_index(drop=True),
        ], axis=1)
        features = features.ffill()
        features = features.fillna(0)

        try:
            results = self.model.predict_proba(features)
        except:
            print(features.tail())
            raise
        y_pred = max(0, results[0][-1] - threshold) # Probs for long
        return 0 if np.isnan(y_pred) else y_pred # long only
