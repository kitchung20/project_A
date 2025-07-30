import pandas as pd


class FeatureBuilder:

    @staticmethod
    def lag_features(series, lags):
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        df = pd.DataFrame()
        for i in range(1, lags + 1):
            df[f"{series.name}_lag_{i}"] = series.shift(i)
        df = df.dropna()
        return df
