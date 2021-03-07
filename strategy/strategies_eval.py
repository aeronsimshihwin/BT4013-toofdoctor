import quantiacsToolbox
import pandas as pd
import utils

def evaluate_by_sharpe(fl, start_date, end_date, interval):
    windows = utils.generate_windows_from(start_date, end_date, interval) # Generate list of in out sample dates
    sharpe_val_lst = []
    eval_date_lst = []
    for window in windows:
        with open('windows.txt', 'w') as filetowrite:
            window_str = str(window)[1:-1]
            filetowrite.write(window_str)
        results = quantiacsToolbox.runts(fl, plotEquity=False)
        sharpe = results["stats"]["sharpe"]
        sharpe_val_lst.append(sharpe)
        eval_date_lst.append(results["evalDate"])
    
    res = pd.DataFrame(windows, columns = ['Start', 'End'])
    res['Sharpe'] = sharpe_val_lst
    res["evalDate"] = eval_date_lst
    return res