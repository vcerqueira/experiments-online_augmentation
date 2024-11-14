import os

import pandas as pd

from utils.load_data.base import LoadDataset


class MiscDataset(LoadDataset):
    DATASET_PATH = 'assets/datasets/'
    DATASET_NAME = 'Misc'

    horizons_map = {
        'NN3': 12,
        'AusDemand': 48,
        'AusDemandWeekly': 12,
    }

    frequency_map = {
        'NN3': 12,
        'AusDemand': 48,
        'AusDemandWeekly': 52,
    }

    context_length = {
        'NN3': 24,
        'AusDemand': 60,
        'AusDemandWeekly': 52,
    }

    min_samples = {
        'NN3': 48,
        'AusDemand': 100,
        'AusDemandWeekly': 52,
    }

    frequency_pd = {
        'NN3': 'ME',
        'AusDemand': '30min',
        'AusDemandWeekly': 'W',
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    @classmethod
    def load_data(cls, group, min_n_instances=None):
        assert group in cls.data_group

        if group == 'NN3':
            fname = 'nn3.csv'
        else:
            fname = 'australian_electricity_demand_dataset.csv'

        df = pd.read_csv(os.path.join(cls.DATASET_PATH, fname))
        df['ds'] = pd.to_datetime(df['ds'])

        if group == 'AusDemandWeekly':
            df_grouped = df.groupby('unique_id').resample('W', on='ds')
            df_resampled = df_grouped.apply({'unique_id': 'first', 'y': 'sum'})
            df = df_resampled.reset_index('ds').reset_index(drop=True)

        return df
