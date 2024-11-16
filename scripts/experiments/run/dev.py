import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast

from metaforecast.utils.data import DataUtils
from metaforecast.synth.callbacks import OnlineDataAugmentationCallback
from utils.workflows.callback import Counter

from utils.load_data.config import DATASETS
from utils.config import (MODELS,
                          MODEL_CONFIG,
                          SYNTH_METHODS,
                          SYNTH_METHODS_PARAMS,
                          EXPERIMENTS_DATASETS,
                          BATCH_SIZE)

# data_name, group = 'Gluonts', 'm1_quarterly'
# data_name, group = 'Misc', 'NN3'
# data_name, group = 'Gluonts', 'm1_monthly'
data_name, group = 'Gluonts', 'nn5_weekly'
# data_name, group = 'Gluonts', 'electricity_weekly'
# data_name, group = 'Misc', 'AusDemandWeekly'

MODEL = 'NHITS'
TSGEN = 'SeasonalMBB'

# pick a synth. progressively increase strength of transform

# LOADING DATA AND SETUP

## LOADING DATA

n_reps = EXPERIMENTS_DATASETS[(data_name, group)]
data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, min_n_instances=min_samples)
batch_size = BATCH_SIZE[data_name, group]

print(df['unique_id'].value_counts())
print(df.shape)

## PREPARING CONFIGS

n_uids = df['unique_id'].nunique()
max_len = df['unique_id'].value_counts().max()
min_len = df['unique_id'].value_counts().min()

input_data = {'input_size': n_lags, 'h': horizon, 'batch_size': batch_size}

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

model_conf = {**input_data, **MODEL_CONFIG.get(MODEL)}

tsgen_params = {k: v for k, v in augmentation_params.items() if k in SYNTH_METHODS_PARAMS[TSGEN]}

tsgen = SYNTH_METHODS[TSGEN](**tsgen_params)

## SPLITS AND MODELS

train, test = DataUtils.train_test_split(df, horizon)

augmentation_cb = OnlineDataAugmentationCallback(generator=tsgen)

models = [MODELS[MODEL](**model_conf,
                        callbacks=[augmentation_cb, Counter(what="batches")],
                        alias=f'{MODEL}(OTF_{TSGEN})')]

models[0].callbacks

# using original train

nf = NeuralForecast(models=models, freq=freq_str)
nf.fit(df=train, val_size=horizon)

fcst = nf.predict()

#
