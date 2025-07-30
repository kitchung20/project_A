import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.gofplots import qqplot

class Visualizer:
    TITLE_SIZE = 16
    LABEL_SIZE = 16
    TICK_SIZE = 16
    plt.rcParams.update({
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': LABEL_SIZE,
        'xtick.labelsize': TICK_SIZE,
        'ytick.labelsize': TICK_SIZE,
        'legend.fontsize': LABEL_SIZE,
        'font.size': LABEL_SIZE,
    })

    @staticmethod
    def plot_hist(series, bins = 50, kde = True):
        series.plot.hist(bins=bins)
        plt.title(f"Histogram of {series.name}")
        plt.xlabel(series.name)
        plt.ylabel("Count")
        plt.show()

    @staticmethod
    def plot_timeseries(series):
        series.plot()
        plt.title(f"Time Series of {series.name}")
        plt.xlabel("Time")
        plt.ylabel(series.name)
        plt.show()

    @staticmethod
    def plot_qq(residuals, title = 'QQ Plot'):
        qqplot(residuals, line='s')
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_pr(pr_df, title = 'Precision–Recall Curve'):
        plt.plot(pr_df['recall'], pr_df['precision'], marker='o')
        plt.title(title)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_pr_grid(models_pr, window_days, save_path):
        n_models = len(models_pr)
        n_wins   = len(window_days)
        fig, axes = plt.subplots(n_models, n_wins,
                                 figsize=(n_wins * 5, n_models * 4),
                                 squeeze=False)
        for i, (model_name, pr_dict) in enumerate(models_pr.items()):
            for j, wd in enumerate(window_days):
                ax = axes[i][j]
                pr_df = pr_dict[wd]
                ax.plot(pr_df['recall'], pr_df['precision'], marker='o')
                ax.set_title(f"{model_name} ±{wd} day window")
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.grid(True)
        plt.tight_layout()
        fig.savefig(save_path)
        plt.show()    