from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.numpy import smape
from datasetsforecast.evaluation import accuracy

from metaforecast.utils.data import DataUtils
from metaforecast.synth.callbacks import OnlineDataAugmentationCallback
from metaforecast.synth import SeasonalMBB

from utils.load_data.config import DATASETS

data_name, group = 'Gluonts', 'nn5_weekly'

data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, min_n_instances=min_samples)

print(df['unique_id'].value_counts())
print(df.shape)

train, test = DataUtils.train_test_split(df, horizon)

tsgen = SeasonalMBB(seas_period=freq_int)

augmentation_cb = OnlineDataAugmentationCallback(generator=tsgen)
# create apriori pipeline
# https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html


models = [NHITS(input_size=horizon,
                h=horizon,
                start_padding_enabled=True,
                accelerator='mps'),
          NHITS(input_size=horizon,
                h=horizon,
                start_padding_enabled=True,
                accelerator='mps',
                callbacks=[augmentation_cb])]

nf = NeuralForecast(models=models, freq=freq_str)

nf.fit(df=train)

fcst = nf.predict()
fcst = fcst.rename(columns={'NHITS1': 'NHITS(MBB)'})

fcst.head()

test = test.merge(fcst.reset_index(), on=['unique_id', 'ds'], how="left")

evaluation_df = accuracy(test, [smape], agg_by=['unique_id'])

eval_df = evaluation_df.drop(columns=['metric', 'unique_id'])

eval_df.mean()
