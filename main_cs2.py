import os
import pickle
import numpy as np
import pandas as pd
from data_loader import DataLoader
from features import FeatureBuilder
from ground_truth import load_ground_truth
from models.arima_model import ARIMAModel
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from evaluation import compute_pr_curve
from visualization import Visualizer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Config
DATA_PATH   = 'data/orbital_elements/CryoSat-2.csv'
GT_FILE     = 'data/manoeuvres/cs2man.txt'
FEATURE     = 'Brouwer mean motion'
SAT_NAME    = 'CryoSat-2'
SAVE_DIR    = 'saved_model'
LAGS        = 3
WINDOW_SIZE = 5  # initial window for sequence building
TEST_SIZE   = 0.2
MATCH_DAYS  = 5

os.makedirs(SAVE_DIR, exist_ok=True)

# Load and preprocess
dl = DataLoader(DATA_PATH, FEATURE)
series = dl.load()
series = dl.preprocess()

# Lagged features for XGBoost
dag = FeatureBuilder.lag_features(series, LAGS)
y = series.loc[dag.index]

# Train/test split by rows
n = len(dag)
split = int(n * (1 - TEST_SIZE))
X_train, y_train = dag.iloc[:split], y.iloc[:split]
X_test, y_test   = dag.iloc[split:], y.iloc[split:]

# ARIMA
print('Running ARIMA')
arima = ARIMAModel(series, SAT_NAME, FEATURE, save_dir=SAVE_DIR)
arima.tune(range(6), range(2), range(6))
arima.fit()
p_arima, r_arima = arima.predict()
anom_arima = pd.Series(np.abs(r_arima), index=r_arima.index)

# XGBoost
print('Running XGBoost')
xgb = XGBoostModel(SAT_NAME, FEATURE, save_dir=SAVE_DIR)
xgb.tune(X_train, y_train, X_test, y_test)
xgb.fit(dag, y)
p_xgb, r_xgb = xgb.predict(dag, y)
anom_xgb = pd.Series(np.abs(r_xgb), index=dag.index[xgb.window+1:])

# Prepare sequences for LSTM
print('Running LSTM')
vals = series.values.reshape(-1,1)
X_seq, y_seq, times = [], [], []
for i in range(WINDOW_SIZE, len(vals)):
    X_seq.append(vals[i-WINDOW_SIZE:i,0])
    y_seq.append(vals[i,0])
    times.append(series.index[i])
X_seq = np.array(X_seq).reshape(-1, WINDOW_SIZE, 1)
y_seq = np.array(y_seq)
# split sequences
tn = len(X_seq)
ss = int(tn * (1 - TEST_SIZE))
X_tr_seq, y_tr_seq = X_seq[:ss], y_seq[:ss]
X_val_seq, y_val_seq = X_seq[ss:], y_seq[ss:]

lstm = LSTMModel(SAT_NAME, FEATURE, save_dir=SAVE_DIR)
lstm.tune(X_tr_seq, y_tr_seq, X_val_seq, y_val_seq)
lstm.fit(X_seq, y_seq)
p_lstm, r_lstm = lstm.predict(X_seq, y_seq)
win = lstm.best_params['window']
anom_lstm = pd.Series(np.abs(r_lstm), index=times[win:])

# Compute PR curves and visualize
_, gt_times = load_ground_truth(GT_FILE)
wdays = [1,5,7]
models = {'ARIMA': anom_arima, 'XGBoost': anom_xgb, 'LSTM': anom_lstm}
prs = {m: {w: compute_pr_curve(w, gt_times, anoms, num_thresholds=50) for w in wdays}
       for m, anoms in models.items()}
os.makedirs('results', exist_ok=True)
#Visualizer.plot_pr_grid(prs, wdays, save_path=f'results/{SAT_NAME}_{FEATURE}_pr.png')
Visualizer.plot_pr_comparison(prs, wdays, save_path=f'results/{SAT_NAME}_{FEATURE}_pr.png')

