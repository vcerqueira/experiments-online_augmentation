import os.path
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
from utils.config import (MODEL,
                          TSGEN,
                          MODELS,
                          MODEL_CONFIG,
                          SYNTH_METHODS,
                          SYNTH_METHODS_PARAMS,
                          REPS_BY_SERIES,
                          BATCH_SIZE)

# LOADING DATA AND SETUP
for data_name, group in REPS_BY_SERIES:

    fp = f'assets/results/{data_name},{group},{MODEL},{TSGEN}.csv'

    if os.path.exists(fp):
        continue

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

    input_data = {'input_size': n_lags,  'h': horizon}

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

    model_conf = {**input_data, **model_params, 'batch_size': batch_size}
    model_conf_2xbatch = {**input_data, **model_params, 'batch_size': batch_size*2}

    tsgen_params = {k: v for k, v in augmentation_params.items() if k in SYNTH_METHODS_PARAMS[TSGEN]}

    tsgen = SYNTH_METHODS[TSGEN](**tsgen_params)

    # SPLITS AND MODELS
    train, test = DataUtils.train_test_split(df, horizon)

    augmentation_cb = OnlineDataAugmentationCallback(generator=tsgen)

    models = [MODELS[MODEL](**model_conf, alias='Original'),
              MODELS[MODEL](**model_conf,
                            callbacks=[augmentation_cb],
                            alias='Online')]
    models_da = [MODELS[MODEL](**model_conf_2xbatch, alias=f'Offline')]
    models_da_max = [MODELS[MODEL](**model_conf_2xbatch, alias=f'OfflineMax')]

    # using original train

    nf = NeuralForecast(models=models, freq=freq_str)
    nf.fit(df=train, val_size=horizon)

    fcst = nf.predict()

    stats_models = [SeasonalNaive(season_length=freq_int)]
    sf = StatsForecast(models=stats_models, freq=freq_str, n_jobs=1)

    sf.fit(train)
    sf_fcst = sf.predict(h=horizon)

    # using augmented train
    apriori_tsgen = SYNTH_METHODS[TSGEN](**tsgen_params)
    train_synth = pd.concat([apriori_tsgen.transform(train) for i in range(n_reps)]).reset_index(drop=True)
    train_ext = pd.concat([train, train_synth]).reset_index(drop=True)

    nf_da = NeuralForecast(models=models_da, freq=freq_str)
    nf_da.fit(df=train_ext, val_size=horizon)
    fcst_ext = nf_da.predict()

    # using max augmented train
    apriori_tsgen = SYNTH_METHODS[TSGEN](**tsgen_params)
    train_synth_max = pd.concat([apriori_tsgen.transform(train)
                                 for i in range(model_params['max_steps'])]).reset_index(drop=True)
    train_ext_max = pd.concat([train, train_synth_max]).reset_index(drop=True)

    nf_da_max = NeuralForecast(models=models_da_max, freq=freq_str)
    nf_da_max.fit(df=train_ext_max, val_size=horizon)
    fcst_extmax = nf_da_max.predict()

    # test set and evaluate

    test = test.merge(fcst.reset_index(), on=['unique_id', 'ds'], how="left")
    test = test.merge(fcst_ext.reset_index(), on=['unique_id', 'ds'], how="left")
    test = test.merge(fcst_extmax.reset_index(), on=['unique_id', 'ds'], how="left")
    test = test.merge(sf_fcst.reset_index(), on=['unique_id', 'ds'], how="left")
    # evaluation_df = evaluate(test, [partial(mase, seasonality=freq_int)], train_df=train)
    evaluation_df = evaluate(test, [partial(mase, seasonality=freq_int), smape], train_df=train)
    # evaluation_df = evaluate(test, [smape], train_df=train)

    evaluation_df.to_csv(fp)

    eval_df = evaluation_df.drop(columns=['metric', 'unique_id'])

    print(eval_df.mean())
    print(eval_df.apply(lambda x: x[x > x.quantile(.95)].mean()))
