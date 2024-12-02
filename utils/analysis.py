import os
import re

import pandas as pd
import plotnine as p9

from utils.config import RESULTS_DIR

DS_MAPPER = {
    'Gluonts-m1_monthly': 'M1-M',
    'Gluonts-m1_quarterly': 'M1-Q',
    'M3-Monthly': 'M3-M',
    'M3-Quarterly': 'M3-Q',
    'Tourism-Monthly': 'T-M',
    'Tourism-Quarterly': 'T-Q',
}

THEME = p9.theme_538(base_family='Palatino', base_size=12) + \
        p9.theme(plot_margin=.025,
                 panel_background=p9.element_rect(fill='white'),
                 plot_background=p9.element_rect(fill='white'),
                 legend_box_background=p9.element_rect(fill='white'),
                 strip_background=p9.element_rect(fill='white'),
                 legend_background=p9.element_rect(fill='white'),
                 axis_text_x=p9.element_text(size=9, angle=0),
                 axis_text_y=p9.element_text(size=9),
                 legend_title=p9.element_blank())


def to_latex_tab(df, round_to_n, rotate_cols: bool):
    if rotate_cols:
        df.columns = [f'\\rotatebox{{90}}{{{x}}}' for x in df.columns]

    annotated_res = []
    for i, r in df.round(round_to_n).iterrows():
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

    return text_tab


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
