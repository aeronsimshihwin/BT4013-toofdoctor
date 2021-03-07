futuresList = ['F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL','F_CT','F_DX','F_EC',
    'F_ED','F_ES','F_FC','F_FV','F_GC','F_HG','F_HO','F_JY','F_KC','F_LB','F_LC','F_LN',
    'F_MD','F_MP','F_NG','F_NQ','F_NR','F_O','F_OJ','F_PA','F_PL','F_RB','F_RU','F_S',
    'F_SB','F_SF','F_SI','F_SM','F_TU','F_TY','F_US','F_W','F_XX','F_YM','F_AX','F_CA',
    'F_DT','F_UB','F_UZ','F_GS','F_LX','F_SS','F_DL','F_ZQ','F_VX','F_AE','F_BG','F_BC',
    'F_LU','F_DM','F_AH','F_CF','F_DZ','F_FB','F_FL','F_FM','F_FP','F_FY','F_GX','F_HP',
    'F_LR','F_LQ','F_ND','F_NY','F_PQ','F_RR','F_RF','F_RP','F_RY','F_SH','F_SX','F_TR',
    'F_EB','F_VF','F_VT','F_VW','F_GD','F_F']

cashList = ['CASH']

futuresAllList = futuresList + cashList

def load_futures_data(data_dir='tickerData'):
    """Simulates data passed to models in main.py"""
    import pandas as pd
    data = {}
    for future in futuresList:
        df = pd.read_csv(f'{data_dir}/{future}.txt')
        df.columns = df.columns.str.strip()
        df.index = pd.to_datetime(df['DATE'], format='%Y%m%d')
        data[future] = df
    return data
