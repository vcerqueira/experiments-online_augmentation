from neuralforecast.models import NHITS
from metaforecast.synth import (SeasonalMBB,
                                Jittering,
                                Scaling,
                                MagnitudeWarping,
                                TimeWarping,
                                DBA,
                                TSMixup)

MODELS = {
    'NHITS': NHITS
}

MODEL_CONFIG = {
    'NHITS': {
        'start_padding_enabled': True,
        'accelerator': 'mps',
        # 'accelerator': 'cpu',
        # 'max_steps': 2000,
        'max_steps': 100,
        'enable_checkpointing': True,
        'early_stop_patience_steps': 50,
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
