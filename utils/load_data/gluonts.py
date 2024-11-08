from pprint import pprint

import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset , dataset_names

from utils.load_data.base import LoadDataset


# pprint(dataset_names)

class GluontsDataset(LoadDataset):
    # ['constant',
    #  'exchange_rate',
    #  'solar-energy',
    #  'electricity',
    #  'traffic',
    #  'exchange_rate_nips',
    #  'electricity_nips',
    #  --'traffic_nips',
    #  'solar_nips',
    #  'wiki2000_nips',
    #  'wiki-rolling_nips',
    #  'taxi_30min',
    #  'kaggle_web_traffic_with_missing',
    #  'kaggle_web_traffic_without_missing',
    #  --'kaggle_web_traffic_weekly',
    #  'm1_yearly',
    #  'm1_quarterly',
    #  'm1_monthly',
    #  'nn5_daily_with_missing',
    #  'nn5_daily_without_missing',
    #  --'nn5_weekly',
    #  --'tourism_monthly',
    #  --'tourism_quarterly',
    #  --'tourism_yearly',
    #  'cif_2016',
    #  'london_smart_meters_without_missing',
    #  'wind_farms_without_missing',
    #  'car_parts_without_missing',
    #  'dominick',
    #  'fred_md',
    #  'pedestrian_counts',
    #  'hospital',
    #  'covid_deaths',
    #  'kdd_cup_2018_without_missing',
    #  'weather',
    #  --'m3_monthly',
    #  --'m3_quarterly',
    #  --'m3_yearly',
    #  --'m3_other',
    #  --'m4_hourly',
    #  --'m4_daily',
    #  --'m4_weekly',
    #  --'m4_monthly',
    #  --'m4_quarterly',
    #  --'m4_yearly',
    #  'm5',
    #  --'uber_tlc_daily',
    #  --'uber_tlc_hourly',
    #  'airpassengers',
    #  'australian_electricity_demand',
    #  'electricity_hourly',
    #  --'electricity_weekly',
    #  'rideshare_without_missing', error
    #  'saugeenday',
    #  'solar_10_minutes',
    #  --'solar_weekly',
    #  'sunspot_without_missing',
    #  'temperature_rain_without_missing', too big
    #  --'vehicle_trips_without_missing',
    #  'ercot', too big
    #  'ett_small_15min', too big
    #  --'ett_small_1h']
    DATASET_NAME = 'GLUONTS'

    horizons_map = {
        'nn5_weekly': 12,
        'electricity_weekly': 12,
        'solar_weekly': 12,
        'kaggle_web_traffic_weekly': 12,
        'australian_electricity_demand': 12,
        'electricity_hourly': 24,
        'uber_tlc_hourly': 24,
        'ett_small_1h': 24,
        'traffic_nips': 24,
        'm1_quarterly': 2,
        'm1_monthly': 8,
        'uber_tlc_daily': 14,
        'vehicle_trips_without_missing': 14,
    }

    # horizons_map_list = {
    #     'electricity_hourly': [48],
    #     'm1_quarterly': [1,2,4,6],
    #     'm1_monthly': [1,6,12,18],
    # }

    frequency_map = {
        'nn5_weekly': 52,
        'electricity_weekly': 52,
        'australian_electricity_demand': 52,
        'solar_weekly': 4,
        'kaggle_web_traffic_weekly': 4,
        'electricity_hourly': 24,
        'uber_tlc_hourly': 24,
        'ett_small_1h': 24,
        'traffic_nips': 24,
        'm1_quarterly': 4,
        'm1_monthly': 12,
        'uber_tlc_daily': 7,
        'vehicle_trips_without_missing': 7,
    }

    context_length = {
        'nn5_weekly': 52,
        'electricity_weekly': 52,
        'australian_electricity_demand': 52,
        'solar_weekly': 12,
        'kaggle_web_traffic_weekly': 12,
        'electricity_hourly': 24,
        'uber_tlc_hourly': 24,
        'ett_small_1h': 24,
        'traffic_nips': 24,
        'm1_quarterly': 4,
        'm1_monthly': 12,
        'uber_tlc_daily': 21,
        'vehicle_trips_without_missing': 21,
    }

    min_samples = {
        'm1_quarterly': 22,
        'm1_monthly': 52,
        'nn5_weekly': 52,
        'australian_electricity_demand': 52,
        'electricity_weekly': 52,
        'kaggle_web_traffic_weekly': 100,
        'traffic_nips': 100,
    }

    frequency_pd = {
        'nn5_weekly': 'W',
        'electricity_weekly': 'W',
        'australian_electricity_demand': 'W',
        'solar_weekly': 'W',
        'kaggle_web_traffic_weekly': 'W',
        'electricity_hourly': 'H',
        'uber_tlc_hourly': 'H',
        'ett_small_1h': 'H',
        'traffic_nips': 'H',
        'm1_quarterly': 'Q',
        'm1_monthly': 'M',
        'uber_tlc_daily': 'D',
        'vehicle_trips_without_missing': 'D',
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    @classmethod
    def load_data(cls,
                  group,
                  min_n_instances=None):
        # group = 'solar_weekly'
        dataset = get_dataset(group, regenerate=False)
        # dataset = get_dataset('australian_electricity_demand', regenerate=True)
        train_list = dataset.train

        df_list = []
        for i, series in enumerate(train_list):
            s = pd.Series(
                series["target"],
                index=pd.date_range(
                    start=series["start"].to_timestamp(),
                    freq=series["start"].freq,
                    periods=len(series["target"]),
                ),
            )

            if group == 'australian_electricity_demand':
                s = s.resample('W').sum()

            s_df = s.reset_index()
            s_df.columns = ['ds', 'y']
            s_df['unique_id'] = f'ID{i}'

            df_list.append(s_df)

        df = pd.concat(df_list).reset_index(drop=True)
        df = df[['unique_id', 'ds', 'y']]

        if min_n_instances is not None:
            df = cls.prune_df_by_size(df, min_n_instances)

        return df


# GluontsDataset.load_data()


