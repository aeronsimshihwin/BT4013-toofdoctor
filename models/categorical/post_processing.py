import numpy as np

THRESHOLD = 0.5

def sign(x, threshold=THRESHOLD):
    return np.sign(x - threshold)

def magnitude(x, threshold=THRESHOLD):
    return x - threshold