import os
import re

import pandas as pd

from utils.config import RESULTS_DIR


DS_MAPPER = {
    'Gluonts-m1_monthly': 'M1-M',
    'Gluonts-m1_quarterly': 'M1-Q',
    'M3-Monthly': 'M3-M',
    'M3-Quarterly': 'M3-Q',
    'Tourism-Monthly': 'T-M',
    'Tourism-Quarterly': 'T-Q',
}


def read_results(metric: str):
    files = os.listdir(RESULTS_DIR)

    results_list = []
    for file in files:
        df_ = pd.read_csv(f'{RESULTS_DIR}/{file}')
        df_['dataset'] = file
        results_list.append(df_)

    res = pd.concat(results_list)
    res = res.query(f'metric=="{metric}"')
    res = res.reset_index(drop=True)
    res[['ds', 'model', 'operation']] = res['dataset'].str.split(',').apply(lambda x: pd.Series(x))
    res = res.drop(columns=['dataset', 'metric', 'unique_id', 'Unnamed: 0'])
    res['operation'] = res['operation'].apply(lambda x: re.sub('.csv', '', x))
    res['ds'] = res['ds'].map(DS_MAPPER)

    return res
