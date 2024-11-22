from neuralforecast.models import NHITS, MLP
from metaforecast.synth import (SeasonalMBB,
                                Jittering,
                                Scaling,
                                MagnitudeWarping,
                                TimeWarping,
                                DBA,
                                TSMixup)

MODELS = {
    'NHITS': NHITS,
    'MLP': MLP
}

MODEL = 'NHITS'
TSGEN = 'TSMixup'
# TSGEN = 'TimeWarping'
# TSGEN = 'TSMixup'

MAX_STEPS = {
    ('Gluonts', 'm1_quarterly'): 1000,
    ('Gluonts', 'm1_monthly'): 1000,
    ('M3', 'Monthly'): 1000,
    ('M3', 'Quarterly'): 1000,
    ('Tourism', 'Monthly'): 1000,
    ('Tourism', 'Quarterly'): 1000,
    # ('Misc', 'NN3'): 500,
    # ('Misc', 'AusDemandWeekly'): 500,
    # ('Gluonts', 'electricity_weekly'): 500,
    # ('Gluonts', 'nn5_weekly'): 500,
}

MODEL_CONFIG = {
    'NHITS': {
        'start_padding_enabled': False,
        'accelerator': 'mps',
        # 'windows_batch_size': 512,
        'scaler_type': 'standard',
        # 'accelerator': 'cpu',
        # 'max_steps': 500,
        # 'val_check_steps': 25,
        # 'enable_checkpointing': True,
        # 'early_stop_patience_steps': 2,
    },
    'MLP': {
        'start_padding_enabled': True,
        'accelerator': 'mps',
        # 'accelerator': 'cpu',
        # 'max_steps': 500,
        'val_check_steps': 50,
        'enable_checkpointing': True,
        'early_stop_patience_steps': 1,
    },
}

SYNTH_METHODS = {
    'SeasonalMBB': SeasonalMBB,
    'Jittering': Jittering,
    'Scaling': Scaling,
    'TSMixup': TSMixup,
    'TimeWarping': TimeWarping,
    'MagnitudeWarping': MagnitudeWarping,
    'DBA': DBA,
}

SYNTH_METHODS_PARAMS = {
    'SeasonalMBB': ['seas_period'],
    'Jittering': [],
    'Scaling': [],
    'MagnitudeWarping': [],
    'TimeWarping': [],
    'DBA': ['max_n_uids'],
    'TSMixup': ['max_n_uids', 'max_len', 'min_len']
}

REPS_BY_SERIES = {
    # ('Gluonts', 'nn5_weekly'): 10,
    # ('Gluonts', 'electricity_weekly'): 10,
    ('Gluonts', 'm1_monthly'): 10,
    ('Gluonts', 'm1_quarterly'): 10,
    # ('Misc', 'NN3'): 10,
    # ('Misc', 'AusDemandWeekly'): 200,
    ('M3', 'Monthly'): 10,
    ('M3', 'Quarterly'): 10,
    ('Tourism', 'Monthly'): 10,
    ('Tourism', 'Quarterly'): 10,
}

BATCH_SIZE = {
    ('M3', 'Monthly'): 32,
    ('M3', 'Quarterly'): 32,
    ('Tourism', 'Monthly'): 32,
    ('Tourism', 'Quarterly'): 32,
    ('Gluonts', 'm1_monthly'): 32,
    ('Gluonts', 'm1_quarterly'): 32,
    # ('Gluonts', 'nn5_weekly'): 16,
    # ('Gluonts', 'electricity_weekly'): 32,
    # ('Misc', 'NN3'): 16,
    # ('Misc', 'AusDemandWeekly'): 2,
}

SYNTH_METHODS_PARAM_VALUES = {
    'SeasonalMBB': {'log': [True, False],
                    'seas_period_multiplier': [.5, 1, 2]},
    'Jittering': {'sigma': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    'Scaling': {'sigma': [0.03, 0.05, 0.1, 0.15, 0.2, 0.3]},
    'MagnitudeWarping': {'sigma': [0.05, 0.1, 0.15], 'knot': [3, 4, 5]},
    'TimeWarping': {'sigma': [0.05, 0.1, 0.15], 'knot': [3, 4, 5]},
    'DBA': {'max_n_uids': [7, 10, 15],
            'dirichlet_alpha': [1.0, 1.5, 2.0],
            'max_iter': [10]},
    'TSMixup': {'max_n_uids': [5, 7, 10],
                'dirichlet_alpha': [1.0, 1.5, 2.0, 3.0]}
}
