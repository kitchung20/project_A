
import sys
import pandas as pd
import numpy as np
from data_loader import DataLoader
from features import FeatureBuilder
from ground_truth import load_ground_truth
from evaluation import compute_pr_curve
from models.arima_model import ARIMAModel
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
import warnings
warnings.filterwarnings("ignore")


FEATURE     = 'Brouwer mean motion'
LAGS        = 3
WINDOW_SIZE = 5       
TEST_SIZE   = 0.2
MATCH_DAYS  = 5

sats = [
    {
        'name': 'CryoSat-2',
        'data_path': 'data/orbital_elements/CryoSat-2.csv',
        'gt_file':   'data/manoeuvres/cs2man.txt'
    },
    {
        'name': 'SARAL',
        'data_path': 'data/orbital_elements/SARAL.csv',
        'gt_file':   'data/manoeuvres/srlman.txt'
    },
    {
        'name': 'Sentinel-3A',
        'data_path': 'data/orbital_elements/Sentinel-3A.csv',
        'gt_file':   'data/manoeuvres/s3aman.txt'
    }
]

results = []

for sat in sats:
    print(f"\n=== Processing {sat['name']} ===")

    dl = DataLoader(sat['data_path'], FEATURE)
    series = dl.load()            
    series = dl.preprocess()     

    _, gt_times = load_ground_truth(sat['gt_file'])

    arima = ARIMAModel(series, sat['name'], FEATURE)
    arima.tune(range(6), range(2), range(6))
    arima.fit()
    _, resid_a = arima.predict()
    anom_a = np.abs(resid_a)

    dag = FeatureBuilder.lag_features(series, LAGS)
    y   = series.loc[dag.index]
    split_idx = int(len(dag) * (1 - TEST_SIZE))
    X_train, y_train = dag.iloc[:split_idx], y.iloc[:split_idx]
    X_test,  y_test  = dag.iloc[split_idx:], y.iloc[split_idx:]
    xgb = XGBoostModel(sat['name'], FEATURE)
    xgb.tune(X_train, y_train, X_test, y_test)
    xgb.fit(dag, y)
    _, resid_x = xgb.predict(dag, y)
    anom_x = pd.Series(np.abs(resid_x), index=dag.index[xgb.window + 1:])

    vals = series.values.reshape(-1, 1)
    X_seq, y_seq, times = [], [], []
    for i in range(WINDOW_SIZE, len(vals)):
        X_seq.append(vals[i-WINDOW_SIZE:i, 0])
        y_seq.append(vals[i, 0])
        times.append(series.index[i])
    X_seq = np.array(X_seq).reshape(-1, WINDOW_SIZE, 1)
    y_seq = np.array(y_seq)

    tn = len(X_seq)
    ss = int(tn * (1 - TEST_SIZE))
    X_tr, y_tr = X_seq[:ss], y_seq[:ss]
    X_val, y_val = X_seq[ss:], y_seq[ss:]

    lstm = LSTMModel(sat['name'], FEATURE)
    lstm.tune(X_tr, y_tr, X_val, y_val)
    lstm.fit(X_seq, y_seq)
    _, resid_l = lstm.predict(X_seq, y_seq)
    win = lstm.best_params.get('window', WINDOW_SIZE)
    anom_l = pd.Series(np.abs(resid_l), index=times[win:])

    for model_name, anoms in [('ARIMA', anom_a), ('XGBoost', anom_x), ('LSTM', anom_l)]:
        pr = compute_pr_curve(MATCH_DAYS, gt_times, anoms)
        pr['f1'] = 2 * pr.precision * pr.recall / (pr.precision + pr.recall)
        best = pr.loc[pr.f1.idxmax()]
        print(f"{model_name:<8}  thr={best.threshold:.4f}  prec={best.precision:.3f}  rec={best.recall:.3f}  f1={best.f1:.3f}")
        results.append({
            'Satellite': sat['name'],
            'Model':     model_name,
            'Threshold': best.threshold,
            'Precision': best.precision,
            'Recall':    best.recall,
            'F1 Score':  best.f1
        })

df = pd.DataFrame(results).set_index(['Satellite','Model'])
print("\n=== All Results ===")
print(df)
