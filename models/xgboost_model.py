
import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

class XGBoostModel:
    def __init__(self, sat_name, feature_name, save_dir='saved_model'):
        self.sat = sat_name; self.feature = feature_name; self.dir=save_dir
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, f'xgboost_{self.sat}_{self.feature}.pkl')
        self.model=None; self.scaler=None; self.window=None; self.best_params=None
        self.wins=[5,10,20]
        self.param_grid={'n_estimators':[100,200],'max_depth':[3,5]}

    def _win(self,data,w):
        X=[]; y=[]
        for i in range(w,len(data)):
            X.append(data[i-w:i]); y.append(data[i])
        return np.array(X),np.array(y)

    def tune(self, Xtr, ytr, Xte, yte):
        if os.path.exists(self.path):
            self.model,self.scaler,self.window,self.best_params = pickle.load(open(self.path,'rb'))
            return self.model
        d = ytr.diff().dropna()
        sc = StandardScaler().fit(d.values.reshape(-1,1))
        sd = sc.transform(d.values.reshape(-1,1)).ravel()
        best=float('inf'); cfg=None
        for w in self.wins:
            Xw,yw=self._win(sd,w)
            grid=GridSearchCV(XGBRegressor(),param_grid=self.param_grid,cv=5, scoring='neg_mean_squared_error')
            grid.fit(Xw,yw)
            scv=-grid.best_score_
            if scv<best: best, cfg = scv, (w,grid.best_params_)
        self.window, self.best_params, self.scaler = cfg[0], cfg[1], sc
        return self.model

    def fit(self, dag, y):
        if os.path.exists(self.path):
            self.model,self.scaler,self.window,self.best_params = pickle.load(open(self.path,'rb'))
            return self.model
        d=y.diff().dropna(); sd=self.scaler.transform(d.values.reshape(-1,1)).ravel()
        Xf,yf=self._win(sd,self.window)
        m=XGBRegressor(**self.best_params); m.fit(Xf,yf)
        self.model=m
        with open(self.path,'wb') as f: pickle.dump((m,self.scaler,self.window,self.best_params),f)
        return m

    def predict(self, dag, y):
        d=y.diff().dropna(); sd=self.scaler.transform(d.values.reshape(-1,1)).ravel()
        Xf,yf=self._win(sd,self.window)
        preds=self.model.predict(Xf); res=yf-preds
        return preds, res