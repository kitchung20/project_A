import os
import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import itertools
import pandas as pd

class ARIMAModel:
    def __init__(self, series, sat_name, feature_name, save_dir='saved_model'):
        self.series = series.copy()
        self.raw = series.copy()
        self.order = None
        self.model = None
        self.mean = None
        self.std = None
        self.sat = sat_name
        self.feature = feature_name
        self.dir = save_dir
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, f'arima_{self.sat}_{self.feature}.pkl')

    def tune(self, p_range, d_range, q_range):
        if os.path.exists(self.path):
            data = pickle.load(open(self.path,'rb'))
            self.model, self.mean, self.std = data[:3]
            self.order = data[3] if len(data)>=4 else self.model.model.order
            return self.order
        s = self.series.copy()
        d = 0
        while d<5 and s.diff().dropna().shape[0]>0:
            s = s.diff().dropna(); d+=1
        import itertools
        res=[]
        for p,d_ext,q in itertools.product(p_range,d_range,q_range):
            try:
                ordt=(p, d+d_ext, q)
                aic = ARIMA(self.series, order=ordt).fit().aic
                res.append((ordt,aic))
            except: pass
        self.order = min(res, key=lambda x:x[1])[0]
        return self.order

    def fit(self):
        if os.path.exists(self.path):
            data = pickle.load(open(self.path,'rb'))
            self.model, self.mean, self.std = data[:3]
            self.order = data[3] if len(data)>=4 else self.model.model.order
            return self.model
        self.mean = self.raw.mean(); self.std = self.raw.std()
        norm = (self.raw - self.mean)/self.std
        self.model = ARIMA(norm, order=self.order).fit()
        with open(self.path,'wb') as f:
            pickle.dump((self.model, self.mean, self.std, self.order), f)
        return self.model

    def predict(self):
        start, end = self.raw.index[1], self.raw.index[-1]
        p_norm = self.model.predict(start=start,end=end)
        p = p_norm*self.std + self.mean
        r = self.raw.iloc[1:] - p
        return p, r