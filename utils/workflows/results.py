import numpy as np
import pandas as pd
from neuralforecast.losses.numpy import smape, rmae


class ResultAnalysis:
    METADATA = ['unique_id', 'ds', 'y', 'horizon']

    METRICS = {
        'smape': smape,
        'rmae': rmae,
    }

    def __init__(self, metric: str):
        self.metric = metric

    def overall_score(self, df):
        models = self.get_model_names(df)

        sc = {}
        for k in models:
            if self.metric == 'rmae':
                data = {
                    'y': df['y'],
                    'y_hat1': df[k],
                    'y_hat2': df['SeasonalNaive'],
                }
            else:
                data = {
                    'y': df['y'],
                    'y_hat': df[k],
                }

            sc[k] = self.METRICS[self.metric](**data)

        sc = pd.Series(sc)
        return sc

    def score_by_group(self, df, group_col: str):
        models = self.get_model_names(df)

        scores = {}
        for group, df_ in df.groupby(group_col):

            err = {}
            for k in models:
                if self.metric == 'rmae':
                    data = {
                        'y': df_['y'],
                        'y_hat1': df_[k],
                        'y_hat2': df_['SeasonalNaive'],
                    }
                else:
                    data = {
                        'y': df_['y'],
                        'y_hat': df_[k],
                    }

                err[k] = self.METRICS[self.metric](**data)

            scores[group] = err

        group_sc = pd.DataFrame(scores).T

        return group_sc

    @staticmethod
    def exp_shortfall(scores: pd.DataFrame, thr: float):
        return scores.apply(lambda x: x[x > x.quantile(thr)].mean())

    @classmethod
    def get_model_names(cls, df):
        return [x for x in df.columns if x not in cls.METADATA]

    @staticmethod
    def map_forecasting_horizon_col(cv):
        if 'cutoff' in cv.columns:
            cv_g = cv.groupby(['unique_id', 'cutoff'])
        else:
            cv_g = cv.groupby(['unique_id'])

        horizon = []
        for g, df in cv_g:
            df = df.sort_values('ds')
            h = np.asarray(range(1, df.shape[0] + 1))
            hs = {
                'horizon': h,
                'ds': df['ds'].values,
                'unique_id': df['unique_id'].values,
            }
            if 'cutoff' in df.columns:
                hs['cutoff'] = df['cutoff'].values,

            hs = pd.DataFrame(hs)
            horizon.append(hs)

        horizon = pd.concat(horizon)

        if 'cutoff' in cv.columns:
            cv = cv.merge(horizon, on=['unique_id', 'ds', 'cutoff'])
        else:
            cv = cv.merge(horizon, on=['unique_id', 'ds'])

        return cv
