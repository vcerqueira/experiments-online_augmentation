import pandas as pd
from datasetsforecast.long_horizon import LongHorizon

from utils.load_data.base import LoadDataset


class LongHorizonDataset(LoadDataset):
    DATASET_NAME = 'lhorizon'
    DATASET_PATH = '/Users/vcerq/Developer/datasets'

    horizons_map = {
        'ETTm2': (96, 192, 336, 720)[0],
        'ETTm1': (96, 192, 336, 720)[0],
        'ETTh2': (96, 192, 336, 720)[0],
        'ETTh1': (96, 192, 336, 720)[0],
        'ECL': (96, 192, 336, 720)[0],
        'Exchange': (96, 192, 336, 720)[0],
        'TrafficL': (96, 192, 336, 720)[0],
        'ILI': (96, 192, 336, 720)[0],
        'Weather': (96, 192, 336, 720)[0],
    }

    frequency_map = {
        'ETTm2': 24 * 4,
        'ETTm1': 24 * 4,
        'ETTh2': 24,
        'ETTh1': 24,
        'ECL': 24 * 4,
        'Exchange': 7,
        'TrafficL': 24,
        'ILI': 52,
        'Weather': 24 * 6,
    }

    context_length = {
        'ETTm2': 24 * 4,
        'ETTm1': 24 * 4,
        'ETTh2': 24,
        'ETTh1': 24,
        'ECL': 24 * 4,
        'Exchange': 15,
        'TrafficL': 24,
        'ILI': 24,
        'Weather': 24 * 6,
    }

    frequency_pd = {
        'ETTm2': '15T',
        'ETTm1': '15T',
        'ETTh2': 'H',
        'ETTh1': 'H',
        'ECL': '15T',
        'Exchange': 'D',
        'TrafficL': 'H',
        'ILI': 'W',
        'Weather': '10M',
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    @classmethod
    def load_data(cls, group, min_n_instances=None):
        df, _, _ = LongHorizon.load(directory=cls.DATASET_PATH, group=group)
        df['ds'] = pd.to_datetime(df['ds'])

        if min_n_instances is not None:
            df = cls.prune_df_by_size(df, min_n_instances)

        return df
