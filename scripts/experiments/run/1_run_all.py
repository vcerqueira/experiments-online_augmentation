import os
from functools import partial

import numpy as np
from neuralforecast import NeuralForecast
from utilsforecast.losses import mase, smape
from utilsforecast.evaluation import evaluate
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast
from metaforecast.utils.data import DataUtils
from metaforecast.synth.callbacks import OnlineDataAugmentation

from utils.load_data.config import DATASETS
from utils.config import MODELS, MODEL_CONFIG, MODEL, SYNTH_METHODS
from utils.load_data.config import DATA_GROUPS
from utils.generators import (get_fixed_online_generator,
                              get_ensemble_online_generator,
                              get_offline_augmented_data,
                              get_offline_augmented_data_rng)

# LOADING DATA AND SETUP
for tsgen in [*SYNTH_METHODS]:

    for data_name, group in DATA_GROUPS:

        fp = f'assets/results/{data_name}-{group},{MODEL},{tsgen}.csv'

        if os.path.exists(fp):
            continue

        data_loader = DATASETS[data_name]
        min_samples = data_loader.min_samples[group]
        df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, min_n_instances=min_samples)

        print(df['unique_id'].value_counts())
        print(df.shape)

        # SPLITS AND MODELS
        train, test = DataUtils.train_test_split(df, horizon)

        # PREPARING CONFIGS
        n_uids = df['unique_id'].nunique()
        max_len = df['unique_id'].value_counts().max() - (2 * horizon)
        min_len = df['unique_id'].value_counts().min() - (2 * horizon)

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

        model_conf = {**input_data, **model_params}
        model_conf_2xbatch = {**input_data, **model_params,
                              'batch_size': model_params['batch_size'] * 2}

        # online augmentation callback setup
        tsgen_default = get_fixed_online_generator(tsgen, augmentation_params)
        tsgen_ensemble = get_ensemble_online_generator(tsgen, freq_int, min_len, max_len)

        oda_def_cb = OnlineDataAugmentation(generator=tsgen_default)
        oda_ens_cb = OnlineDataAugmentation(generator=tsgen_ensemble)

        models = [MODELS[MODEL](**model_conf_2xbatch, alias='Original'),
                  MODELS[MODEL](**model_conf, callbacks=[oda_def_cb], alias='Online(Fixed)'),
                  MODELS[MODEL](**model_conf, callbacks=[oda_ens_cb], alias='Online(Ens)')]

        models_da_1 = [MODELS[MODEL](**model_conf_2xbatch, alias=f'Offline(1))')]
        models_da_10 = [MODELS[MODEL](**model_conf_2xbatch, alias=f'Offline(10))')]
        models_da_eq = [MODELS[MODEL](**model_conf_2xbatch, alias=f'Offline(=)')]
        models_da_rng = [MODELS[MODEL](**model_conf_2xbatch, alias=f'Offline(=,Ens)')]

        # using augmented train
        # offline data augmentation
        n_series_by_uid = int((model_conf['max_steps'] * model_conf['batch_size']) / train['unique_id'].nunique())

        train_da_10 = get_offline_augmented_data(train, tsgen, augmentation_params, 10)
        train_da_1 = get_offline_augmented_data(train, tsgen, augmentation_params, 1)
        train_da_eq = get_offline_augmented_data(train, tsgen, augmentation_params, n_series_by_uid)
        train_da_eq_rng = get_offline_augmented_data_rng(train, tsgen, n_series_by_uid, freq_int, min_len, max_len)

        # using original train
        nf = NeuralForecast(models=models, freq=freq_str)
        nf.fit(df=train, val_size=horizon)
        fcst = nf.predict()

        # seasonal naive baseline
        stats_models = [SeasonalNaive(season_length=freq_int)]
        sf = StatsForecast(models=stats_models, freq=freq_str, n_jobs=1)
        sf.fit(train)
        sf_fcst = sf.predict(h=horizon)

        # training and predicting
        nf_da1 = NeuralForecast(models=models_da_1, freq=freq_str)
        nf_da1.fit(df=train_da_1, val_size=horizon)
        fcst_da1 = nf_da1.predict()

        nf_da10 = NeuralForecast(models=models_da_10, freq=freq_str)
        nf_da10.fit(df=train_da_10, val_size=horizon)
        fcst_da10 = nf_da10.predict()

        nf_da_eq = NeuralForecast(models=models_da_eq, freq=freq_str)
        nf_da_eq.fit(df=train_da_eq, val_size=horizon)
        fcst_da_eq = nf_da_eq.predict()

        nf_da_eqr = NeuralForecast(models=models_da_rng, freq=freq_str)
        nf_da_eqr.fit(df=train_da_eq_rng, val_size=horizon)
        fcst_da_eqr = nf_da_eqr.predict()

        # test set and evaluate

        test = test.merge(fcst.reset_index(), on=['unique_id', 'ds'], how="left")
        test = test.merge(fcst_da1.reset_index(), on=['unique_id', 'ds'], how="left")
        test = test.merge(fcst_da10.reset_index(), on=['unique_id', 'ds'], how="left")
        test = test.merge(fcst_da_eq.reset_index(), on=['unique_id', 'ds'], how="left")
        test = test.merge(fcst_da_eqr.reset_index(), on=['unique_id', 'ds'], how="left")
        test = test.merge(sf_fcst.reset_index(), on=['unique_id', 'ds'], how="left")
        evaluation_df = evaluate(test, [partial(mase, seasonality=freq_int), smape], train_df=train)

        evaluation_df.to_csv(fp)
