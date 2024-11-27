import os

import pandas as pd

from utils.analysis import to_latex_tab
from utils.results import read_results

pd.set_option('display.max_columns', None)

df = read_results('mase')
df.iloc[0]

df.groupby('ds').mean(numeric_only=True)
df.groupby('operation').mean(numeric_only=True)

# method='Jittering'
# res = res.loc[res['dataset'].str.endswith(f'{method}.csv'),:]

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
