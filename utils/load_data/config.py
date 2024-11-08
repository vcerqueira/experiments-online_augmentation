from utils.load_data.m4 import M4Dataset
from utils.load_data.m3 import M3Dataset
from utils.load_data.tourism import TourismDataset
from utils.load_data.nn3 import NN3Dataset
from utils.load_data.gluonts import GluontsDataset
from utils.load_data.long_horizon import LongHorizonDataset

DATASETS = {
    'M4': M4Dataset,
    'M3': M3Dataset,
    'Tourism': TourismDataset,
    'NN3': NN3Dataset,
    'Gluonts': GluontsDataset,
    'lhorizon': LongHorizonDataset,
}

DATA_GROUPS = {
    'M3': ['Monthly', 'Quarterly'],
    'M4': ['Monthly', 'Quarterly'],
    'Tourism': ['Monthly', 'Quarterly'],
    'NN3': ['Monthly'],
    'Gluonts': ['m1_monthly','australian_electricity_demand', 'm1_quarterly', 'nn5_weekly'],
    'lhorizon': [*LongHorizonDataset.horizons_map],
}

