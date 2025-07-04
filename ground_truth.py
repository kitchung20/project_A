import pandas as pd

def load_ground_truth(file_path):
    cols = ['sat','year_start','doy_start','start_h','start_m','year_end','doy_end','end_h','end_m']
    df_gt = pd.read_csv(file_path, delim_whitespace=True, header=None, usecols=range(9), names=cols)
    df_gt['start_date'] = pd.to_datetime(df_gt['year_start']*1000 + df_gt['doy_start'], format='%Y%j')
    df_gt['start_ts'] = df_gt['start_date'] + pd.to_timedelta(df_gt['start_h'], unit='h') + pd.to_timedelta(df_gt['start_m'], unit='m')
    df_gt['end_date'] = pd.to_datetime(df_gt['year_end']*1000 + df_gt['doy_end'], format='%Y%j')
    df_gt['end_ts'] = df_gt['end_date'] + pd.to_timedelta(df_gt['end_h'], unit='h') + pd.to_timedelta(df_gt['end_m'], unit='m')
    df_gt['mid_ts'] = df_gt['start_ts'] + (df_gt['end_ts'] - df_gt['start_ts']) / 2
    return df_gt, pd.DatetimeIndex(df_gt['mid_ts'])