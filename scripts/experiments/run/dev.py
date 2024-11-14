from functools import partial
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from utilsforecast.losses import mase, smape
from utilsforecast.evaluation import evaluate

from metaforecast.utils.data import DataUtils
from metaforecast.synth.callbacks import OnlineDataAugmentationCallback
from utils.workflows.callback import OnlineDACallback

from utils.load_data.config import DATASETS
from utils.config import MODELS, MODEL_CONFIG, SYNTH_METHODS, SYNTH_METHODS_PARAMS

# data_name, group = 'Gluonts', 'nn5_weekly'
# data_name, group = 'Gluonts', 'electricity_weekly'
# data_name, group = 'Gluonts', 'm1_monthly'
# data_name, group = 'Gluonts', 'm1_quarterly'
data_name, group = 'Misc', 'NN3'
# data_name, group = 'Misc', 'AusDemandWeekly'
MODEL = 'NHITS'
TSGEN = 'SeasonalMBB'
N_REPS = 10

# LOADING DATA AND SETUP

## LOADING DATA

data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, min_n_instances=min_samples)

print(df['unique_id'].value_counts())
print(df.shape)

## PREPARING CONFIGS

n_uids = df['unique_id'].nunique()
max_len = df['unique_id'].value_counts().max()
min_len = df['unique_id'].value_counts().min()

input_data = {'input_size': n_lags, 'h': horizon, }

max_n_uids = int(np.round(np.log(n_uids), 0))
max_n_uids = 2 if max_n_uids < 2 else max_n_uids

augmentation_params = {
    'seas_period': freq_int,
    'max_n_uids': max_n_uids,
    'max_len': max_len,
    'min_len': min_len,
}

model_conf = {**input_data, **MODEL_CONFIG.get(MODEL), 'batch_size': n_uids}

tsgen_params = {k: v for k, v in augmentation_params.items()
                if k in SYNTH_METHODS_PARAMS[TSGEN]}

tsgen = SYNTH_METHODS[TSGEN](**tsgen_params)

## SPLITS AND MODELS

train, test = DataUtils.train_test_split(df, horizon)

augmentation_cb = OnlineDACallback(generator=tsgen)
da_metaf_cb = OnlineDataAugmentationCallback(generator=tsgen)

models = [MODELS[MODEL](**model_conf,
                        callbacks=[augmentation_cb],
                        alias=f'{MODEL}(Custom_{TSGEN})'),
          MODELS[MODEL](**model_conf,
                        callbacks=[da_metaf_cb],
                        alias=f'{MODEL}(OTF_{TSGEN})'),
          ]

# using original train

nf = NeuralForecast(models=models, freq=freq_str)
nf.fit(df=train, val_size=horizon)

fcst = nf.predict()

# test set and evaluate

test = test.merge(fcst.reset_index(), on=['unique_id', 'ds'], how="left")
evaluation_df = evaluate(test, [partial(mase, seasonality=freq_int)], train_df=train)
# evaluation_df = evaluate(test, [partial(mase, seasonality=freq_int), smape], train_df=train)
# evaluation_df = evaluate(test, [smape], train_df=train)

eval_df = evaluation_df.drop(columns=['metric', 'unique_id'])

print(eval_df.mean())
print(eval_df.apply(lambda x: x[x > x.quantile(.95)].mean()))
