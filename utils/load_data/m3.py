from datasetsforecast.m3 import M3

from utils.load_data.base import LoadDataset


class M3Dataset(LoadDataset):
    DATASET_NAME = 'M3'

    horizons_map = {
        'Quarterly': 8,
        'Monthly': 12,
        'Yearly': 4,
    }

    # horizons_map_list = {
    #     'Quarterly': [1, 2, 4, 6],
    #     'Monthly': [1, 6, 12, 18],
    # }

    frequency_map = {
        'Quarterly': 4,
        'Monthly': 12,
        'Yearly': 1,
    }

    context_length = {
        'Quarterly': 8,
        'Monthly': 24,
        'Yearly': 3,
    }

    min_samples = {
        'Quarterly': 100,
        'Monthly': 130,
        'Yearly': 10,
    }

    frequency_pd = {
        'Quarterly': 'Q',
        'Monthly': 'ME',
        'Yearly': 'Y',
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    @classmethod
    def load_data(cls, group, min_n_instances=None):
        ds, *_ = M3.load(cls.DATASET_PATH, group=group)

        if min_n_instances is not None:
            ds = cls.prune_df_by_size(ds, min_n_instances)

        return ds
