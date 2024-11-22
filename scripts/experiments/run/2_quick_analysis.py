import os

import pandas as pd

RESULTS_DIR = 'assets/results'
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

files = os.listdir(RESULTS_DIR)

results_list = []
for file in files:
    # if file == 'Gluonts,nn5_weekly,NHITS,SeasonalMBB.csv':
    #     continue
    df_ = pd.read_csv(f'{RESULTS_DIR}/{file}')
    df_['dataset'] = file

    results_list.append(df_)

res = pd.concat(results_list)

res_metric = res.query('metric=="mase"')
eval_df = res_metric.drop(columns=['metric', 'unique_id', 'Unnamed: 0'])

# eval_df['tsg'] = eval_df['dataset'].str.split(',').apply(lambda x: x[-1])
# eval_df['ds'] = eval_df['dataset'].str.split(',').apply(lambda x: '_'.join(x[:2]))
# eval_df = eval_df.drop(columns='dataset')

# eval_df.groupby(['ds','tsg']).mean()
# eval_df.groupby(['ds']).median(numeric_only=True).mean()
# eval_df.groupby(['tsg']).median(numeric_only=True).mean()

print(eval_df.groupby('dataset').mean().T.mean(axis=1))
print(eval_df.groupby('dataset').median().T.mean(axis=1))
print(eval_df.groupby('dataset').median().T.median(axis=1))
print(eval_df.groupby('dataset').median().T.rank().T.mean())
print(eval_df.groupby('dataset').mean().T.rank().T.mean())

eval_df.groupby('dataset').mean().reset_index(drop=True)
