import pandas as pd
import plotnine as p9

from utils.analysis import to_latex_tab, read_results, THEME

df = read_results('mase')

df = df[['Online(Fixed)', 'Online(Ens)', 'Offline(1))',
         'Offline(10))', 'Offline(=)', 'Offline(=,Ens)', 'Original', 'SeasonalNaive', 'ds',
         'model', 'operation']]

COLUMN_MAP = {
    'Online(Fixed)': 'Online',
    'Online(Ens)': 'Online(E)',
    'Offline(1))': 'Offline(1)',
    'Offline(10))': 'Offline(10)',
    'Offline(=,Ens)': 'Offline(=,E)',
    'SeasonalNaive': 'Naive',
}

OPERATION_MAP = {
    'MagnitudeWarping': 'M-Warp',
    'TimeWarping': 'T-Warp',
    'SeasonalMBB': 'MBB',
}

COLORS = [
    '#1e3d59',  # Deep navy blue
    '#457b9d',  # Steel blue
    '#a8dadc',  # Light blue
    '#7da87b',  # Sage green
    '#e07a5f',  # Terracotta
    '#b5446e',  # Deep rose
    '#9b2226'  # Deep red
]

APPROACH_COLORES = [
    '#2c3e50',  # Dark slate blue
    '#34558b',  # Royal blue
    '#4b7be5',  # Bright blue
    '#6db1bf',  # Light teal
    '#bf9b7a',  # Warm tan
    '#d17f5e',  # Warm coral
    '#c44536',  # Burnt orange red
    '#8b1e3f'   # Deep burgundy
]

df = df.rename(columns=COLUMN_MAP)
df['operation'] = df['operation'].replace(OPERATION_MAP)

# df.groupby('ds').mean(numeric_only=True)
# df.groupby('operation').mean(numeric_only=True)

# overall details on table
perf_by_all = df.groupby(['ds', 'model', 'operation']).mean(numeric_only=True)
perf_by_mod_ds = df.groupby(['model', 'operation']).mean(numeric_only=True)
avg_score = perf_by_mod_ds.mean().values
avg_rank = perf_by_all.rank(axis=1).mean().round(2).values

perf_by_mod_ds.loc[('All', 'Average'), :] = avg_score
perf_by_mod_ds.loc[('All', 'Avg. Rank'), :] = avg_rank

tex_tab = to_latex_tab(perf_by_mod_ds, 4, rotate_cols=True)
print(tex_tab)

# grouped bar plot
# x=operation, y= average score, group=model
scores = df.groupby(['model', 'operation']).mean(numeric_only=True)['Online']
scores_df = scores.reset_index()
scores_df.columns = ['Model', 'Method', 'MASE']

plot = \
    p9.ggplot(data=scores_df,
              mapping=p9.aes(x='Model',
                             y='MASE',
                             fill='Method')) + \
    p9.geom_bar(position='dodge',
                stat='identity',
                width=0.9) + \
    THEME + \
    p9.theme(axis_title_y=p9.element_text(size=12),
             axis_title_x=p9.element_blank(),
             axis_text=p9.element_text(size=12)) + \
    p9.scale_fill_manual(values=COLORS)

plot.save('mase_by_model_op.pdf', height=5, width=12)

#

ds_perf = df.groupby(['ds', 'operation']).mean(numeric_only=True)['Online'].reset_index()
ds_perf.columns = ['Dataset', 'Method', 'MASE']

plot = \
    p9.ggplot(data=ds_perf,
              mapping=p9.aes(x='Method',
                             y='MASE',
                             fill='Method')) + \
    p9.facet_wrap('~Dataset', nrow=2) + \
    p9.geom_bar(position='dodge',
                stat='identity',
                width=0.9) + \
    THEME + \
    p9.theme(axis_title_y=p9.element_text(size=12),
             axis_title_x=p9.element_blank(),
             axis_text=p9.element_text(size=12),
             axis_text_x=p9.element_text(angle=60)) + \
    p9.scale_fill_manual(values=COLORS)

plot.save('mase_by_model_ds.pdf', height=7, width=12)

#

ds_perf = df.groupby(['ds']).mean(numeric_only=True).drop(columns='Offline(=,E)').reset_index().melt('ds')
ds_perf.columns = ['Dataset', 'Approach', 'MASE']

plot = \
    p9.ggplot(data=ds_perf,
              mapping=p9.aes(x='Approach',
                             y='MASE',
                             fill='Approach')) + \
    p9.facet_wrap('~Dataset', nrow=2) + \
    p9.geom_bar(position='dodge',
                stat='identity',
                width=0.9) + \
    THEME + \
    p9.theme(axis_title_y=p9.element_text(size=12),
             axis_title_x=p9.element_blank(),
             axis_text=p9.element_text(size=12),
             axis_text_x=p9.element_text(angle=60)) + \
    p9.scale_fill_manual(values=APPROACH_COLORES)

plot.save('mase_by_approach_ds.pdf', height=7, width=12)


# effectiveness

df_eff = df.groupby(['ds', 'operation', 'model']).mean(numeric_only=True)

effectiveness = \
    pd.concat({c: df_eff[c] < df_eff['Original']
               for c in df_eff.columns if c not in ['Original', 'Naive']}, axis=1)

effect_df = effectiveness.mean().reset_index()
effect_df.columns = ['Method', 'Effectiveness']
effect_df['Method'] = pd.Categorical(effect_df['Method'], categories=effect_df['Method'])

plot = \
    p9.ggplot(data=effect_df,
              mapping=p9.aes(x='Method',
                             y='Effectiveness')) + \
    p9.geom_bar(position='dodge',
                stat='identity',
                width=0.9,
                fill='#0b3b24') + \
    THEME + \
    p9.theme(axis_title_y=p9.element_text(size=14),
             axis_text=p9.element_text(size=13)) + \
    p9.labs(x='')

plot.save('effectiveness.pdf', width=10, height=4)
