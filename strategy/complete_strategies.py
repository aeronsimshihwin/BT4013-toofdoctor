import numpy as np
import pandas as pd

def futures_only(futures_only_pos_vector):
    '''
    only invest in futures
    '''
    if np.sum(np.abs(futures_only_pos_vector)) != 0:
        cash_frac = 0.0
        complete_pos_vector = np.array([*futures_only_pos_vector, cash_frac])
        return complete_pos_vector
    else:
        cash_frac = 1.0
        complete_pos_vector = np.array([*futures_only_pos_vector, cash_frac])
        return complete_pos_vector

def futures_hold(futures_only_pos_vector, previous_position):
    if np.sum(np.abs(futures_only_pos_vector)) != 0:
        complete_pos_vector = np.array([*futures_only_pos_vector, 0.0])
        return complete_pos_vector
    else:
        return previous_position

def cash_and_futures(futures_only_pos_vector):
    '''
    % cash = % futures that are weighted zero (i.e. not long or short)
    '''
    frac_weighted_zero = (1 - np.count_nonzero(futures_only_pos_vector)) / len(futures_only_pos_vector)
    frac_cash = (sum(futures_only_pos_vector) / (1-frac_weighted_zero)) * frac_weighted_zero
    
    complete_pos_vector = [x for x in futures_only_pos_vector]
    complete_pos_vector.append(frac_cash)

    return complete_pos_vector
