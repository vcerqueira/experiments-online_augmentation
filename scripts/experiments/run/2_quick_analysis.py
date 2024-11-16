import os

import pandas as pd

RESULTS_DIR = 'assets/results'

files = os.listdir(RESULTS_DIR)

results_list = []
for file in files:
    df_ = pd.read_csv(f'{RESULTS_DIR}/{file}')
    df_['dataset'] = file

    results_list.append(df_)

res = pd.concat(results_list)

res_metric = res.query('metric=="smape"')
eval_df = res_metric.drop(columns=['metric', 'unique_id', 'Unnamed: 0'])

print(eval_df.groupby('dataset').mean().T.mean(axis=1))
print(eval_df.groupby('dataset').mean().T.rank().T.mean())
print(eval_df.groupby('dataset').apply(lambda x: x[x > x.quantile(.9)].mean().T).mean())

eval_df.groupby('dataset').mean().reset_index(drop=True)
