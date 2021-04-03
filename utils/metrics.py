import numpy as np
import math

def opportunity_cost(y_actual, y_pred, cost):
    mask = (y_actual != y_pred).apply(lambda x: int(x))
    return sum(mask * abs(cost))

def simple_moving_average(x):
    return x.mean()

def exponential_moving_average(x):
    return x.ewm(com=0.5).mean().iloc[-1]