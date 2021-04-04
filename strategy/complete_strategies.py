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
        # print(complete_pos_vector)
        return complete_pos_vector
    else:
        return previous_position

def futures_subset(futures_only_pos_vector, subset_csv):
    '''
    trade only futures with positive sharpe on validation set
    '''
    if np.sum(np.abs(futures_only_pos_vector)) == 0:
        complete_pos_vector = np.array([*futures_only_pos_vector, 1.0])
        return complete_pos_vector
    else:
        subset_df = pd.read_csv(subset_csv)
        trade_mask = list(subset_df.trade)
        trade_pos = [x[0] * x[1] for x in zip(futures_only_pos_vector, trade_mask)]
        complete_pos_vector = np.array([*trade_pos, 0.0])
        return complete_pos_vector

def cash_and_futures(futures_only_pos_vector):
    '''
    % cash = % futures that are weighted zero (i.e. not long or short)
    '''
    frac_weighted_zero = 1 - np.count_nonzero(futures_only_pos_vector) / len(futures_only_pos_vector) # (1 - np.count_nonzero(futures_only_pos_vector)) / len(futures_only_pos_vector)
    if frac_weighted_zero == 1:
        frac_cash = 1
    else:
        frac_cash = (sum(futures_only_pos_vector) / (1-frac_weighted_zero)) * frac_weighted_zero
    
    complete_pos_vector = [x for x in futures_only_pos_vector]
    complete_pos_vector.append(frac_cash)

    return np.array(complete_pos_vector)
