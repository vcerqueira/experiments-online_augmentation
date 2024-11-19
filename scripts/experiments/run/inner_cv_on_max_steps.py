from functools import partial
from neuralforecast import NeuralForecast
from utilsforecast.losses import mase
from utilsforecast.evaluation import evaluate
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast

from metaforecast.utils.data import DataUtils

from utils.load_data.config import DATASETS
from utils.config import (MODELS,
                          MODEL_CONFIG,
                          REPS_BY_SERIES,
                          BATCH_SIZE, MODEL)

# data_name, group = 'Gluonts', 'nn5_weekly'
# data_name, group = 'Gluonts', 'm1_quarterly'
# data_name, group = 'Misc', 'NN3'
# data_name, group = 'Gluonts', 'm1_monthly'
# data_name, group = 'Gluonts', 'electricity_weekly'
# data_name, group = 'Misc', 'AusDemandWeekly'
# data_name, group = 'M3', 'Monthly'
data_name, group = 'M3', 'Quarterly'

MAX_STEPS_LIST = [100, 150, 250, 500, 750, 1000, 1500]

## LOADING DATA

n_reps = REPS_BY_SERIES[(data_name, group)]
data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, min_n_instances=min_samples)
batch_size = BATCH_SIZE[data_name, group]

input_data = {'input_size': n_lags, 'h': horizon}

# SPLITS AND MODELS
train, _ = DataUtils.train_test_split(df, horizon)

models = [MODELS[MODEL](**MODEL_CONFIG.get(MODEL), **input_data, max_steps=i, alias=f'MS_{i}')
          for i in MAX_STEPS_LIST]

# using original train

nf = NeuralForecast(models=models, freq=freq_str)
cv_nf = nf.cross_validation(df=train, val_size=horizon)

stats_models = [SeasonalNaive(season_length=freq_int)]
sf = StatsForecast(models=stats_models, freq=freq_str, n_jobs=1)

cv_sf = sf.cross_validation(df=train, h=horizon)

# test set and evaluate

cv = cv_nf.merge(cv_sf.reset_index().drop(columns=['y']), on=['unique_id', 'ds', 'cutoff'])

evaluation_df = evaluate(df=cv.drop(columns=['cutoff']),
                         metrics=[partial(mase, seasonality=freq_int)],
                         train_df=train)

eval_df = evaluation_df.drop(columns=['metric', 'unique_id'])

print(eval_df.mean())
