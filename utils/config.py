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
# TSGEN = 'Scaling'
TSGEN = 'SeasonalMBB'
# TSGEN = 'Jittering'

MODEL_CONFIG = {
    'NHITS': {
        'start_padding_enabled': False,
        'accelerator': 'mps',
        # 'accelerator': 'cpu',
        'max_steps': 250,
        'val_check_steps': 50,
        'enable_checkpointing': True,
        'early_stop_patience_steps': 5,
    },
    'MLP': {
        'start_padding_enabled': False,
        'accelerator': 'mps',
        # 'accelerator': 'cpu',
        'max_steps': 200,
        # 'max_steps': 10,
        # 'max_steps': 100,
        # 'enable_checkpointing': True,
        'early_stop_patience_steps': 3,
    },
}

SYNTH_METHODS = {
    'SeasonalMBB': SeasonalMBB,
    'Jittering': Jittering,
    'Scaling': Scaling,
    'MagnitudeWarping': MagnitudeWarping,
    'TimeWarping': TimeWarping,
    'DBA': DBA,
    'TSMixup': TSMixup,
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
    ('Gluonts', 'nn5_weekly'): 10,
    ('M3', 'Monthly'): 10,
    ('M3', 'Quarterly'): 10,
    ('Gluonts', 'electricity_weekly'): 10,
    ('Gluonts', 'm1_monthly'): 10,
    ('Gluonts', 'm1_quarterly'): 10,
    ('Misc', 'NN3'): 10,
    ('Misc', 'AusDemandWeekly'): 200,
}

BATCH_SIZE = {
    ('Gluonts', 'nn5_weekly'): 16,
    ('M3', 'Monthly'): 32,
    ('M3', 'Quarterly'): 32,
    ('Gluonts', 'electricity_weekly'): 32,
    ('Gluonts', 'm1_monthly'): 32,
    ('Gluonts', 'm1_quarterly'): 32,
    ('Misc', 'NN3'): 16,
    ('Misc', 'AusDemandWeekly'): 2,
}
