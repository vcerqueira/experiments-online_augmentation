import os

import pandas as pd

from utils.load_data.base import LoadDataset


class NN3Dataset(LoadDataset):
    DATASET_PATH = 'assets/datasets/'
    DATASET_NAME = 'NN3'

    horizons_map = {
        'Monthly': 12
    }

    frequency_map = {
        'Monthly': 12
    }

    context_length = {
        'Monthly': 24
    }

    min_samples = {
        'Monthly': 48,
    }

    frequency_pd = {
        'Monthly': 'ME'
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    @classmethod
    def load_data(cls, group, min_n_instances=None):
        assert group in cls.data_group

        df = pd.read_csv(os.path.join(cls.DATASET_PATH, 'nn3.csv'))
        df['ds'] = pd.to_datetime(df['ds'])

        return df
