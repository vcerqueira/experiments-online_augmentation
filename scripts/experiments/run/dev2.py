import pandas as pd
from neuralforecast import NeuralForecast

from metaforecast.utils.data import DataUtils
from utils.workflows.callback import TestCallback

from utils.load_data.config import DATASETS
from utils.config import (MODELS,
                          MODEL_CONFIG,
                          REPS_BY_SERIES,
                          BATCH_SIZE, MODEL, TSGEN)

# data_name, group = 'Gluonts', 'm1_quarterly'
# data_name, group = 'Misc', 'NN3'
data_name, group = 'Gluonts', 'm1_monthly'

## LOADING DATA

n_reps = REPS_BY_SERIES[(data_name, group)]
data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, min_n_instances=min_samples)
batch_size = BATCH_SIZE[data_name, group]

print(df['unique_id'].value_counts())
print(df.shape)

df_uids = pd.get_dummies(df['unique_id']).astype(int)
df_uid_list = df_uids.columns.tolist()
df_ext = pd.concat([df, df_uids], axis=1)


input_data = {'input_size': n_lags, 'h': horizon}

# SPLITS AND MODELS
train, test = DataUtils.train_test_split(df, horizon)

models = [MODELS[MODEL](**input_data,
                        accelerator='cpu',
                        callbacks=[TestCallback()])]

nf = NeuralForecast(models=models, freq=freq_str)
nf.fit(df=df_ext, val_size=horizon)
