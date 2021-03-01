from .post_processing import *
from .arima import model as arima_wrapper

models = {
    'arima': arima_wrapper,
}