from functools import partial
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from utilsforecast.losses import mase, smape
from utilsforecast.evaluation import evaluate
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast

from metaforecast.utils.data import DataUtils
from metaforecast.synth.callbacks import OnlineDataAugmentationCallback

from utils.load_data.config import DATASETS
from utils.config import (MODELS,
                          MODEL_CONFIG,
                          SYNTH_METHODS,
                          SYNTH_METHODS_PARAMS,
                          REPS_BY_SERIES,
                          BATCH_SIZE, MODEL, TSGEN)
from utils.workflows.callback import OnlineDACallback

# data_name, group = 'Gluonts', 'm1_quarterly'
# data_name, group = 'Misc', 'NN3'
# data_name, group = 'Gluonts', 'm1_monthly'
data_name, group = 'Gluonts', 'nn5_weekly'
# data_name, group = 'Gluonts', 'electricity_weekly'
# data_name, group = 'Misc', 'AusDemandWeekly'


# pick a synth. progressively increase strength of transform

# LOADING DATA AND SETUP

## LOADING DATA

n_reps = REPS_BY_SERIES[(data_name, group)]
data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, min_n_instances=min_samples)
batch_size = BATCH_SIZE[data_name, group]

print(df['unique_id'].value_counts())
print(df.shape)

# PREPARING CONFIGS
n_uids = df['unique_id'].nunique()
max_len = df['unique_id'].value_counts().max()
min_len = df['unique_id'].value_counts().min()

input_data = {'input_size': n_lags, 'h': horizon}

max_n_uids = int(np.round(np.log(n_uids), 0))
max_n_uids = 2 if max_n_uids < 2 else max_n_uids

augmentation_params = {
    'seas_period': freq_int,
    'max_n_uids': max_n_uids,
    'max_len': max_len,
    'min_len': min_len,
}

model_params = MODEL_CONFIG.get(MODEL)
if model_params['start_padding_enabled'] and group == 'AusDemandWeekly':
    model_params['start_padding_enabled'] = False

model_conf = {**input_data, **model_params,
              'batch_size': batch_size}
model_conf_2xbatch = {**input_data, **model_params,
                      'batch_size': batch_size * 2}

tsgen_params = {k: v for k, v in augmentation_params.items() if k in SYNTH_METHODS_PARAMS[TSGEN]}

tsgen = SYNTH_METHODS[TSGEN](**tsgen_params)

# SPLITS AND MODELS
train, test = DataUtils.train_test_split(df, horizon)

augmentation_cb = OnlineDACallback(generator=tsgen,max_steps=model_params['max_steps'])
augmentation_cb2 = OnlineDataAugmentationCallback(generator=tsgen)

models = [MODELS[MODEL](**model_conf_2xbatch,
                  alias='original'),
    MODELS[MODEL](**model_conf_2xbatch,
                  callbacks=[augmentation_cb],
                  alias='Online'),
          MODELS[MODEL](**model_conf,
                        callbacks=[augmentation_cb2],
                        alias='Onlinemf')
          ]
# models_da_max = [MODELS[MODEL](**model_conf_2xbatch, alias=f'OfflineMax')]

# using original train

nf = NeuralForecast(models=models, freq=freq_str)
nf.fit(df=train, val_size=horizon)

fcst = nf.predict()

stats_models = [SeasonalNaive(season_length=freq_int)]
sf = StatsForecast(models=stats_models, freq=freq_str, n_jobs=1)

sf.fit(train)
sf_fcst = sf.predict(h=horizon)

test = test.merge(fcst.reset_index(), on=['unique_id', 'ds'], how="left")
test = test.merge(sf_fcst.reset_index(), on=['unique_id', 'ds'], how="left")
# evaluation_df = evaluate(test, [partial(mase, seasonality=freq_int)], train_df=train)
evaluation_df = evaluate(test, [partial(mase, seasonality=freq_int), smape], train_df=train)
# evaluation_df = evaluate(test, [smape], train_df=train)


eval_df = evaluation_df.drop(columns=['metric', 'unique_id'])

print(eval_df.mean())
print(eval_df.apply(lambda x: x[x > x.quantile(.95)].mean()))
#
# Original         0.508986
# Online           0.459190
# OfflineMax       0.560207
# SeasonalNaive    0.550490
# dtype: float32
# Original         2.099162
# Online           1.905085
# OfflineMax       2.218878
# SeasonalNaive    1.820815
