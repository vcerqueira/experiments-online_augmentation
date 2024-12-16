from neuralforecast.models import NHITS, MLP, KAN
from metaforecast.synth import (SeasonalMBB,
                                Jittering,
                                Scaling,
                                MagnitudeWarping,
                                TimeWarping,
                                DBA,
                                TSMixup)

REPS_BY_SERIES = 10
MODEL = 'MLP'
TSGEN = 'TSMixup'

RESULTS_DIR = 'assets/results'

MODELS = {
    'NHITS': NHITS,
    'MLP': MLP,
    'KAN': KAN,
}

MODEL_CONFIG = {
    'NHITS': {
        'start_padding_enabled': False,
        'accelerator': 'mps',
        # 'accelerator': 'cpu',
        'scaler_type': 'standard',
        'max_steps': 1000,
        'batch_size': 32,
    },
    'MLP': {
        'start_padding_enabled': False,
        'accelerator': 'mps',
        # 'accelerator': 'cpu',
        'scaler_type': 'standard',
        'batch_size': 32,
        'max_steps': 1000,
    },
    'KAN': {
        'accelerator': 'gpu',
        # 'accelerator': 'cpu',
        'scaler_type': 'standard',
        'batch_size': 32,
        'max_steps': 1000,
    },

}

SYNTH_METHODS = {
    'SeasonalMBB': SeasonalMBB,
    'Jittering': Jittering,
    'Scaling': Scaling,
    'TimeWarping': TimeWarping,
    'MagnitudeWarping': MagnitudeWarping,
    'TSMixup': TSMixup,
    'DBA': DBA,
}

SYNTH_METHODS_ARGS = {
    'SeasonalMBB': ['seas_period'],
    'Jittering': [],
    'Scaling': [],
    'MagnitudeWarping': [],
    'TimeWarping': [],
    'DBA': ['max_n_uids'],
    'TSMixup': ['max_n_uids', 'max_len', 'min_len']
}

SYNTH_METHODS_GRID_VALUES = {
    'SeasonalMBB': {'log': [True, False], 'seas_period_multiplier': [.5, 1, 2]},
    'Jittering': {'sigma': [0.03, 0.05, 0.1, 0.15, 0.2, 0.3]},
    'Scaling': {'sigma': [0.03, 0.05, 0.1, 0.15, 0.2, 0.3]},
    'MagnitudeWarping': {'sigma': [0.05, 0.1, 0.15], 'knot': [3, 4, 5]},
    'TimeWarping': {'sigma': [0.05, 0.1, 0.15], 'knot': [3, 4, 5]},
    'DBA': {'max_n_uids': [5, 7, 10, 15], 'dirichlet_alpha': [1.0, 1.5, 2.0]},
    'TSMixup': {'max_n_uids': [5, 7, 10, 15], 'dirichlet_alpha': [1.0, 1.5, 2.0]}
}
