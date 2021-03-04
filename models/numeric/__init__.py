import numpy as np
from .arima import ArimaWrapper, SAVE_DIR

models = [
    ('arima', ArimaWrapper, [], dict(y = lambda data: np.log(data['CLOSE'].tail(40)).diff()), SAVE_DIR)
]