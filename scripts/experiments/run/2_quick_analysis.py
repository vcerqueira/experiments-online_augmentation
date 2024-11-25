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
eval_df.groupby(['tsg','dataset']).mean().reset_index(level='dataset', drop=True)

resdf = eval_df.groupby(['tsg']).mean(numeric_only=True)

annotated_res = []
for i, r in resdf.round(4).iterrows():
    top_2 = r.sort_values().unique()[:2]
    if len(top_2) < 2:
        raise ValueError('only one score')

    best1 = r[r == top_2[0]].values[0]
    best2 = r[r == top_2[1]].values[0]

    r[r == top_2[0]] = f'\\textbf{{{best1}}}'
    r[r == top_2[1]] = f'\\underline{{{best2}}}'

    annotated_res.append(r)

annotated_res = pd.DataFrame(annotated_res).astype(str)

text_tab = annotated_res.to_latex(caption='CAPTION', label='tab:scores_by_ds')
print(text_tab)
#
# print(eval_df.drop(columns='dataset').groupby('tsg').mean().T.mean(axis=1))
# print(eval_df.drop(columns='dataset').groupby('tsg').median().T.mean(axis=1))
# print(eval_df.drop(columns='dataset').groupby('tsg').mean().T.median(axis=1))
# print(eval_df.drop(columns='dataset').groupby('tsg').apply(lambda x: x.rank(axis=1).mean()).mean())
# print(eval_df.drop(columns='dataset').groupby('tsg').median().T.rank().T.mean())
# print(eval_df.drop(columns='dataset').groupby('tsg').mean().T.rank().T.mean())
# print(eval_df.drop(columns='dataset').groupby('tsg').apply(lambda x: x[x > x.quantile(.9)].mean()).mean())
