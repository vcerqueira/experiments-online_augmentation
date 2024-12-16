import numpy as np
import pandas as pd
import plotnine as p9

from utils.load_data.config import DATASETS
from utils.config import SYNTH_METHODS, SYNTH_METHODS_ARGS
from utils.load_data.config import DATA_GROUPS

data_name, group = DATA_GROUPS[1]

data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, min_n_instances=min_samples)

n_uids = df['unique_id'].nunique()
max_len = df['unique_id'].value_counts().max() - (2 * horizon)
min_len = df['unique_id'].value_counts().min() - (2 * horizon)
max_n_uids = int(np.round(np.log(n_uids), 0))
max_n_uids = 2 if max_n_uids < 2 else max_n_uids
augmentation_params = {
    'seas_period': freq_int,
    'max_n_uids': max_n_uids,
    'max_len': max_len,
    'min_len': min_len,
}

uid = 'ID0'

original = df.query(f'unique_id=="{uid}"')
original['unique_id'] = 'Original'

uid_list = [original]

for tsgen in SYNTH_METHODS:
    print(tsgen)
    tsgen_params = {k: v for k, v in augmentation_params.items()
                    if k in SYNTH_METHODS_ARGS[tsgen]}

    if tsgen == 'Jittering':
        mod = SYNTH_METHODS[tsgen](**tsgen_params, sigma=0.15)
    else:
        mod = SYNTH_METHODS[tsgen](**tsgen_params)

    df_synth = mod.transform(df).query(f'unique_id.str.startswith("{uid}_")')
    df_synth['unique_id'] = tsgen

    uid_list.append(df_synth)

aug_df = pd.concat(uid_list)

COLORS = [
    '#34558b',  # Royal blue
    '#4b7be5',  # Bright blue
    '#6db1bf',  # Light teal
    '#bf9b7a',  # Warm tan
    '#d17f5e',  # Warm coral
    '#c44536',  # Burnt orange red
]

plot = \
    p9.ggplot(aug_df) + \
    p9.aes(x='ds',
           y='y',
           group='unique_id',
           color='unique_id') + \
    p9.geom_line(size=1) + \
    p9.theme_538(base_family='Palatino', base_size=12) + \
    p9.theme(plot_margin=.005,
             panel_background=p9.element_rect(fill='white'),
             plot_background=p9.element_rect(fill='white'),
             legend_box_background=p9.element_rect(fill='white'),
             strip_background=p9.element_rect(fill='white'),
             legend_background=p9.element_rect(fill='white'),
             axis_text_x=p9.element_text(size=12, angle=0),
             axis_text_y=p9.element_text(size=12),
             axis_title=p9.element_text(size=12),
             legend_title=p9.element_blank()) + \
    p9.scale_x_datetime(date_breaks='2 years') + \
    p9.labs(x='', y='value', title=f'M1 Monthly, id={uid}') + \
    p9.scale_color_manual(values=COLORS)

plot.save('example_aug.pdf', width=10, height=4)
