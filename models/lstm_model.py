import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

class LSTMModel:
    def __init__(self, sat_name, feature_name, save_dir='saved_model'):
        self.sat=sat_name; self.feature=feature_name; self.dir=save_dir
        os.makedirs(self.dir, exist_ok=True)
        self.path=os.path.join(self.dir,f'lstm_{self.sat}_{self.feature}.pkl')
        self.model=None; self.scaler=None; self.best_params=None
        self.param_grid={ 'window':[5,10,20], 'hidden_dim':[50,100], 'num_layers':[1,2], 'dropout':[0.2,0.5], 'learning_rate':[0.001,0.01], 'batch_size':[32], 'epochs':[10] }

    def _seq(self,data,w):
        X=[]; y=[]
        for i in range(w,len(data)):
            X.append(data[i-w:i]); y.append(data[i])
        return np.array(X),np.array(y)

    def tune(self,Xtr, ytr, Xval, yval):
        if os.path.exists(self.path):
            self.model,self.scaler,self.best_params=pickle.load(open(self.path,'rb'))
            return self.model
        raw=np.concatenate([ytr,yval]); d=np.diff(raw,prepend=raw[0])
        sc=StandardScaler().fit(d.reshape(-1,1)); sd=sc.transform(d.reshape(-1,1)).ravel()
        best=float('inf'); cfg=None
        for p in ParameterGrid(self.param_grid):
            Xw,yw=self._seq(sd,p['window'])
            m=Sequential()
            for l in range(p['num_layers']-1): m.add(LSTM(p['hidden_dim'],return_sequences=True,input_shape=(p['window'],1)))
            m.add(LSTM(p['hidden_dim'],input_shape=(p['window'],1)))
            m.add(Dropout(p['dropout'])); m.add(Dense(1))
            m.compile(optimizer=Adam(p['learning_rate']),loss='mse')
            hist=m.fit(Xw,yw,epochs=p['epochs'],batch_size=p['batch_size'],validation_split=0.2,verbose=0)
            vl=min(hist.history['val_loss'])
            if vl<best: best, cfg = vl, (p, sc)
        self.best_params, self.scaler = cfg
        return self.model

    def fit(self,X, y):
        if os.path.exists(self.path):
            self.model,self.scaler,self.best_params=pickle.load(open(self.path,'rb'))
            return self.model
        d=np.diff(y,prepend=y[0]); sd=self.scaler.transform(d.reshape(-1,1)).ravel()
        w=self.best_params['window']; Xf,yf=self._seq(sd,w)
        p=self.best_params; m=Sequential()
        for l in range(p['num_layers']-1): m.add(LSTM(p['hidden_dim'],return_sequences=True,input_shape=(w,1)))
        m.add(LSTM(p['hidden_dim'],input_shape=(w,1)))
        m.add(Dropout(p['dropout'])); m.add(Dense(1))
        m.compile(optimizer=Adam(p['learning_rate']),loss='mse')
        m.fit(Xf,yf,epochs=p['epochs'],batch_size=p['batch_size'],verbose=0)
        self.model=m
        with open(self.path,'wb') as f: pickle.dump((m,self.scaler,self.best_params),f)
        return m

    def predict(self,X, y):
        d=np.diff(y,prepend=y[0]); sd=self.scaler.transform(d.reshape(-1,1)).ravel()
        w=self.best_params['window']; Xf,_=self._seq(sd,w)
        preds=self.model.predict(Xf).ravel(); res=d[w:]-preds
        return preds, res
