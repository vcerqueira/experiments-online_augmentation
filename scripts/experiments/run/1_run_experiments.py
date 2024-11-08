from functools import partial
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from utilsforecast.losses import mase, smape
from utilsforecast.evaluation import evaluate
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast

from metaforecast.utils.data import DataUtils
from metaforecast.synth.callbacks import OnlineDataAugmentationCallback
from metaforecast.synth import SeasonalMBB

from utils.load_data.config import DATASETS

# data_name, group = 'Gluonts', 'nn5_weekly'
# data_name, group = 'Gluonts', 'electricity_weekly'
data_name, group = 'Gluonts', 'm1_monthly'
# data_name, group = 'NN3', 'Monthly'

data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, min_n_instances=min_samples)

print(df['unique_id'].value_counts())
print(df.shape)

MODEL_CONFIG = {
    'input_size': n_lags,
    'h': horizon,
    'start_padding_enabled': True,
    'accelerator': 'mps',
    'max_steps': 2000,
    # 'val_check_steps': 30,
    'enable_checkpointing': True,
    'early_stop_patience_steps': 50,
}

# setup

train, test = DataUtils.train_test_split(df, horizon)

tsgen = SeasonalMBB(seas_period=freq_int)
augmentation_cb = OnlineDataAugmentationCallback(generator=tsgen)

models = [NHITS(**MODEL_CONFIG),
          NHITS(**MODEL_CONFIG, callbacks=[augmentation_cb])]
models_da = [NHITS(**MODEL_CONFIG)]

# using original train

nf = NeuralForecast(models=models, freq=freq_str)
nf.fit(df=train, val_size=horizon)

fcst = nf.predict()
fcst = fcst.rename(columns={'NHITS1': 'NHITS(MBB)'})

# sf

stats_models = [SeasonalNaive(season_length=freq_int)]
sf = StatsForecast(models=stats_models, freq=freq_str, n_jobs=1)

sf.fit(train)
sf_fcst = sf.predict(h=horizon)

# using augmented train
apriori_tsgen = SeasonalMBB(seas_period=freq_int)

train_synth = apriori_tsgen.transform(train)

train_ext = pd.concat([train, train_synth]).reset_index(drop=True)

nf2 = NeuralForecast(models=models_da, freq=freq_str)
nf2.fit(df=train_ext, val_size=horizon)

fcst_ext = nf2.predict()
fcst_ext = fcst_ext.rename(columns={'NHITS': 'NHITS(ApMBB)'})

# test set and evaluate

test = test.merge(fcst.reset_index(), on=['unique_id', 'ds'], how="left")
test = test.merge(fcst_ext.reset_index(), on=['unique_id', 'ds'], how="left")
test = test.merge(sf_fcst.reset_index(), on=['unique_id', 'ds'], how="left")
# evaluation_df = evaluate(test, [partial(mase, seasonality=52)], train_df=train)
evaluation_df = evaluate(test, [partial(mase, seasonality=52), smape], train_df=train)
# evaluation_df = evaluate(test, [smape], train_df=train)

eval_df = evaluation_df.drop(columns=['metric', 'unique_id'])

print(eval_df.mean())
print(eval_df.apply(lambda x: x[x > x.quantile(.95)].mean()))