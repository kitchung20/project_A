import pandas as pd
class DataLoader:
    def __init__(self, path, obs_feature):
        self.path = path
        self.obs_feature = obs_feature
        self.df = None
        self.series = None

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.path, index_col=0, parse_dates=True)
        return self.df

    def preprocess(self) -> pd.Series:
        df = self.df.copy()
        obs = (df[self.obs_feature] - df[self.obs_feature].mean()) / df[self.obs_feature].std()
        obs.name = self.obs_feature
        self.series = obs
        return self.series