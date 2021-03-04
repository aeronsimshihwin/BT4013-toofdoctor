import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression

import utils
from model_validation import walk_forward

SAVED_DIR = "/saved_models/categorical/logreg"
FUTURES_LIST = utils.futuresList

# model.fit(X)
# class model:
#     initialise()
#       X_vars = ""
#       y_var = ""
#       cost = ""
#     fit()
#     predict()
#         predict
#         process predict
#         return predict
