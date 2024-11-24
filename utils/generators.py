from typing import List, Dict
from itertools import product

import numpy as np
import pandas as pd

from utils.config import (SYNTH_METHODS,
                          SYNTH_METHODS_ARGS,
                          SYNTH_METHODS_GRID_VALUES)


def get_all_combinations(config_space: Dict) -> List[Dict]:
    keys, values = zip(*config_space.items())

    all_combinations = []
    for combination in product(*values):
        config = dict(zip(keys, combination))
        all_combinations.append(config)

    return all_combinations


def get_param_combinations(generator_name, freq_int, min_len, max_len):
    sample_pars = SYNTH_METHODS_GRID_VALUES[generator_name]

    if 'seas_period_multiplier' in sample_pars:
        sample_pars['seas_period'] = [int(freq_int * x) for x in sample_pars['seas_period_multiplier']]
        sample_pars.pop('seas_period_multiplier')

    if generator_name == 'TSMixup':
        sample_pars['min_len'] = [min_len]
        sample_pars['max_len'] = [max_len]

    sample_params_comb = get_all_combinations(sample_pars)

    return sample_params_comb


def get_fixed_online_generator(generator_name, augmentation_params):
    tsgen_params = {k: v for k, v in augmentation_params.items()
                    if k in SYNTH_METHODS_ARGS[generator_name]}

    generator_default = SYNTH_METHODS[generator_name](**tsgen_params)

    return generator_default


def get_ensemble_online_generator(generator_name, freq_int, min_len, max_len):
    sample_params_comb = get_param_combinations(generator_name, freq_int, min_len, max_len)

    generator_ensemble = []
    for param_comb in sample_params_comb:
        generator_ensemble.append(SYNTH_METHODS[generator_name](**param_comb))

    return generator_ensemble


def get_offline_augmented_data(train_, generator_name, augmentation_params, n_series_by_uid):
    train = train_.copy()

    tsgen_params = {k: v for k, v in augmentation_params.items()
                    if k in SYNTH_METHODS_ARGS[generator_name]}

    offline_def_tsgen = SYNTH_METHODS[generator_name](**tsgen_params)

    train_synth = pd.concat([offline_def_tsgen.transform(train)
                             for _ in range(n_series_by_uid)]).reset_index(drop=True)

    train_augmented = pd.concat([train, train_synth]).reset_index(drop=True)

    return train_augmented


def get_offline_augmented_data_rng(train_, generator_name, n_series_by_uid, freq_int, min_len, max_len):
    train = train_.copy()

    sample_params_comb = get_param_combinations(generator_name, freq_int, min_len, max_len)

    train_synth_rng_l = []
    for _ in range(n_series_by_uid):
        pars_i = np.random.choice(sample_params_comb)

        tsgen_i = SYNTH_METHODS[generator_name](**pars_i)

        train_synth_rng_l.append(tsgen_i.transform(train))

    train_synth_rng = pd.concat(train_synth_rng_l)

    train_augmented = pd.concat([train, train_synth_rng]).reset_index(drop=True)

    return train_augmented
