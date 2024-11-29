import pandas as pd
import plotnine as p9

from utils.analysis import to_latex_tab
from utils.results import read_results
from utils.plots import THEME

pd.set_option('display.max_columns', None)

df = read_results('mase')

df.groupby('ds').mean(numeric_only=True)
df.groupby('operation').mean(numeric_only=True)

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
scores = df.groupby(['model', 'operation']).mean(numeric_only=True)['Online(Fixed)']
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
             axis_text=p9.element_text(size=12))

plot.save('mase_by_model_op.pdf', height=5, width=12)

#
