import os

import pandas as pd

from utils.analysis import to_latex_tab

RESULTS_DIR = 'assets/results_rest'
pd.set_option('display.max_columns', None)

files = os.listdir(RESULTS_DIR)

results_list = []
for file in files:
    df_ = pd.read_csv(f'{RESULTS_DIR}/{file}')
    df_['dataset'] = file

    results_list.append(df_)

res = pd.concat(results_list)

# method='Jittering'
# res = res.loc[res['dataset'].str.endswith(f'{method}.csv'),:]

res_metric = res.query('metric=="mase"')
eval_df = res_metric.drop(columns=['metric', 'unique_id', 'Unnamed: 0'])

# eval_df['ds'] = eval_df['dataset'].str.split(',').apply(lambda x: '_'.join(x[:2]))
# eval_df = eval_df.drop(columns='dataset')

# eval_df.groupby(['ds','tsg']).mean()
# eval_df.groupby(['ds']).median(numeric_only=True).mean()
# eval_df.groupby(['tsg']).median(numeric_only=True).mean()

print(eval_df.groupby('dataset').mean().T.mean(axis=1))
print(eval_df.groupby('dataset').median().T.mean(axis=1))
print(eval_df.groupby('dataset').mean().T.median(axis=1))
print(eval_df.groupby('dataset').apply(lambda x: x.rank(axis=1).mean()).mean())
print(eval_df.groupby('dataset').median().T.rank().T.mean())
print(eval_df.groupby('dataset').mean().T.rank().T.mean())
print(eval_df.groupby('dataset').apply(lambda x: x[x > x.quantile(.9)].mean()).mean())

eval_df.groupby('dataset').mean().reset_index(drop=True)

eval_df['tsg'] = eval_df['dataset'].str.split(',').apply(lambda x: x[-1])
eval_df.groupby(['tsg', 'dataset']).mean().reset_index(level='dataset', drop=True)

resdf = eval_df.groupby(['tsg']).mean(numeric_only=True)

text_tab = to_latex_tab(resdf, 4)

print(text_tab)
#
# print(eval_df.drop(columns='dataset').groupby('tsg').mean().T.mean(axis=1))
# print(eval_df.drop(columns='dataset').groupby('tsg').median().T.mean(axis=1))
# print(eval_df.drop(columns='dataset').groupby('tsg').mean().T.median(axis=1))
# print(eval_df.drop(columns='dataset').groupby('tsg').apply(lambda x: x.rank(axis=1).mean()).mean())
# print(eval_df.drop(columns='dataset').groupby('tsg').median().T.rank().T.mean())
# print(eval_df.drop(columns='dataset').groupby('tsg').mean().T.rank().T.mean())
# print(eval_df.drop(columns='dataset').groupby('tsg').apply(lambda x: x[x > x.quantile(.9)].mean()).mean())
